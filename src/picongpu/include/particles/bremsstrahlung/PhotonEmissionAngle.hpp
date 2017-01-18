/**
 * Copyright 2016-2017 Heiko Burau
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/cursor/Cursor.hpp"
#include "cuSTL/cursor/navigator/PlusNavigator.hpp"
#include "cuSTL/cursor/tools/LinearInterp.hpp"
#include "cuSTL/cursor/BufferCursor.hpp"
#include "algorithms/math.hpp"
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/odeint/integrate/integrate.hpp>
#include <boost/math/tools/minima.hpp>
#include <limits>
#include <utility>

namespace picongpu
{
namespace particles
{
namespace bremsstrahlung
{

namespace detail
{

/** Functor mapping `delta` to the photon emission polar angle `theta`,
 * where delta is a uniformly distributed random number between zero and one.
 */
struct GetPhotonAngleFunctor
{
    typedef typename ::PMacc::result_of::Functor<
        ::PMacc::cursor::tools::LinearInterp<float_X>,
        ::PMacc::cursor::BufferCursor<float_X, DIM2> >::type LinInterpCursor;

    typedef float_X type;

    LinInterpCursor linInterpCursor;
    float_X lnMinGamma;
    float_X lnMaxGamma;

    /** constructor
     *
     * @param linInterpCursor lookup table for the photon emission angle.
     */
    HDINLINE GetPhotonAngleFunctor(LinInterpCursor linInterpCursor)
        : linInterpCursor(linInterpCursor)
    {
        this->lnMinGamma = math::log(photon::MIN_GAMMA);
        this->lnMaxGamma = math::log(photon::MAX_GAMMA);
    }

    /** Return the polar emission angle of the photon.
     *
     * @param delta uniformly distributed random number between zero and one.
     * @param gamma relativistic factor of the incident electron.
     */
    HDINLINE float_X operator()(const float_X delta, const float_X gamma) const
    {
        const float_X deltaLookupPos = delta * static_cast<float_64>(photon::NUM_SAMPLES_DELTA - 1);

        const float_X lnGamma = algorithms::math::log(gamma);
        const float_X gammaLookupPos =
            (lnGamma - this->lnMinGamma) /
            (this->lnMaxGamma - this->lnMinGamma) *
            static_cast<float_X>(photon::NUM_SAMPLES_GAMMA - 1);

        if (picLog::log_level & picLog::CRITICAL::lvl)
        {
            if(gamma > photon::MAX_GAMMA)
            {
                printf("[Bremsstrahlung] error lookup table: gamma = %g is out of range.\n",
                       gamma);
            }
        }

        return this->linInterpCursor[float2_X(deltaLookupPos, gammaLookupPos)];
    }
};

} // namespace detail

/** Creates and holds the lookup table for the photon emission angle.
 */
struct GetPhotonAngle
{
    typedef detail::GetPhotonAngleFunctor GetPhotonAngleFunctor;

private:

    typedef boost::shared_ptr<PMacc::container::DeviceBuffer<float_X, DIM2> > MyBuf;
    MyBuf dBufTheta;

    /** probability density at polar angle theta.
     * It's the ultrarelativistic limit of the dipole radiation formula, see e.g. Jackson, chap. 15.2
     */
    struct Probability
    {
        const float_64 gamma2;
        Probability(const float_64 gamma) : gamma2(gamma*gamma) {}

        template<typename T_State>
        void operator()(const T_State &p, T_State &dpdtheta, const float_64 theta) const
        {
            const float_64 theta2 = theta*theta;
            const float_64 denom = float_64(1.0) + gamma2 * theta2;

            dpdtheta[0] = float_64(3.0) * theta * gamma2 * (float_64(1.0) + gamma2*gamma2 * theta2*theta2) /
                          (denom*denom*denom*denom);
        }
    };

    /** Return the absolute deviation of a delta, computed from a given theta, and a reference delta.
     *
     * Delta is the angular emission probability (normalized to one) integrated from zero to theta,
     * where theta is the angle between the photon momentum and the final electron momentum.
     */
    struct AimForDelta
    {
        const float_64 targetDelta;
        const float_64 gamma;

        /** constructor
         *
         * @param targetDelta reference delta
         * @param gamma relativistic factor
         */
        AimForDelta(const float_64 targetDelta, const float_64 gamma) :
            targetDelta(targetDelta), gamma(gamma) {}

        float_64 delta(const float_64 theta, const float_64 gamma) const
        {
            namespace odeint = boost::numeric::odeint;

            typedef boost::array<float_64, 1> state_type;

            state_type integral_result = {0.0};
            const float_64 lowerLimit = 0.0;
            const float_64 upperLimit = theta;
            const float_64 stepwidth = (upperLimit - lowerLimit) / float_64(1000.0);
            Probability integrand(gamma);
            odeint::integrate(integrand, integral_result, lowerLimit, upperLimit, stepwidth);

            return integral_result[0];
        }

        float_64 operator()(const float_64 theta) const
        {
            return math::abs(this->delta(theta, this->gamma) - this->targetDelta);
        }
    };

    /** Return the maximal theta which corresponds to the maximal delta and a given gamma
     *
     * @param gamma relativistic factor
     */
    float_64 maxTheta(const float_64 gamma) const
    {
        AimForDelta aimForDelta(photon::MAX_DELTA, gamma);

        std::pair<float_64, float_64> minimum;

        minimum = boost::math::tools::brent_find_minima(
            aimForDelta,
            0.0,
            M_PI,
            std::numeric_limits<float_64>::digits);

        return minimum.first;
    }

    /** computes the polar emission angle theta.
     *
     * @param delta uniformly distributed random number within [0, 1] or (0, 1)
     * @param gamma relativistic factor
     * @param maxTheta maximal theta
     */
    float_64 theta(const float_64 delta, const float_64 gamma, const float_64 maxTheta) const
    {
        AimForDelta aimForDelta(delta, gamma);
        const float_64 minTheta = 0.0;
        std::pair<float_64, float_64> minimum;

        minimum = boost::math::tools::brent_find_minima(
            aimForDelta,
            minTheta,
            maxTheta,
            std::numeric_limits<float_64>::digits);

        return minimum.first;
    }

public:

    /** Generate lookup table
     */
    void init()
    {
        // there is a margin of one cell to make the linear interpolation valid for border cells.
        this->dBufTheta = MyBuf(new PMacc::container::DeviceBuffer<float_X, DIM2>(
            photon::NUM_SAMPLES_DELTA + 1,
            photon::NUM_SAMPLES_GAMMA + 1));

        PMacc::container::HostBuffer<float_X, DIM2> hBufTheta(this->dBufTheta->size());
        hBufTheta.assign(float_X(0.0));
        auto curTheta = hBufTheta.origin();

        const float_64 lnMinGamma = math::log(photon::MIN_GAMMA);
        const float_64 lnMaxGamma = math::log(photon::MAX_GAMMA);

        for(uint32_t gammaIdx = 0; gammaIdx < photon::NUM_SAMPLES_GAMMA; gammaIdx++)
        {
            const float_64 lnGamma_norm = static_cast<float_64>(gammaIdx) /
                                          static_cast<float_64>(photon::NUM_SAMPLES_GAMMA - 1);
            const float_64 gamma = math::exp(lnMinGamma + (lnMaxGamma - lnMinGamma) * lnGamma_norm);
            const float_64 maxTheta = this->maxTheta(gamma);

            for(uint32_t deltaIdx = 0; deltaIdx < photon::NUM_SAMPLES_DELTA; deltaIdx++)
            {
                const float_64 delta = photon::MAX_DELTA * static_cast<float_64>(deltaIdx) /
                                       static_cast<float_64>(photon::NUM_SAMPLES_DELTA - 1);

                *curTheta(deltaIdx, gammaIdx) = static_cast<float_X>(this->theta(delta, gamma, maxTheta));
            }
        }

        *this->dBufTheta = hBufTheta;
    }

    /** Return a functor mapping `delta` to the photon emission polar angle `theta`,
     * where delta is a uniformly distributed random number within [0, 1] or (0, 1)
     */
    GetPhotonAngleFunctor getPhotonAngleFunctor() const
    {
        GetPhotonAngleFunctor::LinInterpCursor linInterpCursor =
            PMacc::cursor::tools::LinearInterp<float_X>()(this->dBufTheta->origin());

        return GetPhotonAngleFunctor(linInterpCursor);
    }
};

} // namespace bremsstrahlung
} // namespace particles
} // namespace picongpu
