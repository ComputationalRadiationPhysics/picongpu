/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/startPosition/RandomImpl.hpp"
#include "picongpu/particles/startPosition/generic/FreeRng.def"

#include <pmacc/algorithms/math/defines/pi.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace startPosition
        {
            namespace acc
            {
                template<typename T_ParamClass>
                struct RandomBinomialImpl : public RandomImpl<T_ParamClass>
                {
                    using Base = RandomImpl<T_ParamClass>;

                    /** Set in-cell position and weighting
                     *
                     * The scheme is described in the .def file.
                     *
                     * @tparam T_Rng functor::misc::RngWrapper, type of the random number generator
                     * @tparam T_Particle pmacc::Particle, particle type
                     *
                     * @param rng random number generator
                     * @param particle particle to be manipulated
                     */
                    template<typename T_Rng, typename T_Particle>
                    HDINLINE void operator()(T_Rng& rng, T_Particle& particle)
                    {
                        // Initialize the weighting here as only here we got random number generator
                        if(!m_isInitialized)
                        {
                            /* This is required for normal distribution to be a good approximation to binomial
                             * (> 5 is minimem, >= 8 is very safe)
                             */
                            PMACC_CASSERT_MSG(
                                __RandomBinomialImpl_numParticlesPerCell_must_be_at_least_8,
                                T_ParamClass::numParticlesPerCell >= 8);
                            auto const s = math::sqrt(static_cast<float_X>(T_ParamClass::numParticlesPerCell));
                            auto const z = getStandardNormal(rng);
                            this->m_weighting /= (1.0_X + z / s);
                            auto const minWeighting = MIN_WEIGHTING;
                            this->m_weighting = math::max(this->m_weighting, minWeighting);
                            m_isInitialized = true;
                        }
                        // The rest is exactly same as for RandomImpl
                        Base::operator()(rng, particle);
                    }

                    // If we initialized the weighting in operator()
                    bool m_isInitialized = false;

                private:
                    /** Get a standard normally distributed random number from a given uniform distribution generator
                     *
                     * For convenience this functor works with uniform distribution, so implement abort
                     * Box-Muller transform manually for this use case.
                     *
                     * @tparam T_UniformRng functor::misc::RngWrapper, type of the uniform random number generator
                     *
                     * @param uniformRng uniform random number generator
                     */
                    template<typename T_UniformRng>
                    HDINLINE auto getStandardNormal(T_UniformRng& uniformRng)
                    {
                        auto u1 = uniformRng();
                        while(!u1)
                            u1 = uniformRng();
                        auto const u2 = uniformRng();
                        auto const z = math::sqrt(-2.0_X * math::log(u1))
                                       * math::cos(pmacc::math::Pi<float_X>::doubleValue * u2);
                        return z;
                    }
                };

            } // namespace acc
        } // namespace startPosition
    } // namespace particles
} // namespace picongpu
