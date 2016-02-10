/**
 * Copyright 2015-2016 Heiko Burau
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

#include "pmacc_types.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/cursor/Cursor.hpp"
#include "cuSTL/cursor/navigator/PlusNavigator.hpp"
#include "cuSTL/cursor/tools/LinearInterp1D.hpp"
#include "cuSTL/cursor/BufferCursor.hpp"
#include "simulation_defines.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/array.hpp>
#include <boost/numeric/odeint/integrate/integrate.hpp>
#include <boost/math/tr1.hpp> /* cyl_bessel_k */

namespace picongpu
{
namespace particles
{
namespace bremsstrahlung
{

namespace detail
{

/** Map `x` to the internal lookup table and return the result of the
 * first or the second synchrotron function for `x`.
 */
struct MapToLookupTable
{
    typedef typename ::PMacc::result_of::Functor<
        ::PMacc::cursor::tools::LinearInterp1D<float_X>,
        ::PMacc::cursor::BufferCursor<float_X, DIM1> >::type LinInterpCursor;

    typedef float_X type;

    LinInterpCursor linInterpCursor;

    /** constructor
     *
     * @param linInterpCursor lookup table of the first or the second
     * synchrotron function.
     */
    HDINLINE MapToLookupTable(LinInterpCursor linInterpCursor)
        : linInterpCursor(linInterpCursor) {}

    /** Returns F_1(x) or F_2(x)

     * @param x position of the synchrotron function to be evaluated
     */
    HDINLINE float_X operator()(const float_X x) const
    {
        /* This mapping increases the sample point density for small values of x
         * where the synchrotron functions have a divergent slope. Without this mapping
         * the emission probabilty of low-energy photons is underestimated.
         *
         * This is the inverse mapping of the mapping in @see:`SynchrotronFunctions::init()`
         */
        const float_X x_m = PMacc::algorithms::math::pow(x, float_X(1.0/3.0));

        const float_X cutOff = static_cast<float_X>(SYNC_FUNCS_CUTOFF);

        if(x_m >= cutOff)
            return float_X(0.0);
        else
            return this->linInterpCursor[x_m / static_cast<float_X>(SYNC_FUNCS_STEP_WIDTH)];
    }
};

typedef ::PMacc::cursor::Cursor<
    MapToLookupTable,
    ::PMacc::cursor::PlusNavigator,
    float_X> SyncFuncCursor;

} // namespace detail


/** Lookup table for synchrotron functions.
 *
 * Provides cursors for the first and the second synchrotron function
 */
class SynchrotronFunctions
{
public:
    typedef detail::SyncFuncCursor SyncFuncCursor;
private:

    typedef boost::shared_ptr<PMacc::container::DeviceBuffer<float_X, DIM1> > MyBuf;
    MyBuf dBuf_SyncFuncs[2]; // two synchrotron functions

    struct BesselK
    {
        template<typename T_State, typename T_Time>
        void operator()(const T_State &x, T_State &dxdt, T_Time t) const
        {
            dxdt[0] = boost::math::tr1::cyl_bessel_k(5.0/3.0, t);
        }
    };

    /** First synchrotron function
     */
    float_64 F_1(const float_64 x) const
    {
        if(x == float_64(0.0))
            return float_64(0.0);

        using namespace boost::numeric::odeint;
        typedef boost::array<float_64, 1> state_type;

        state_type integral_result = {0.0};
        const float_64 upper_bound(SYNC_FUNCS_F1_INTEGRAL_BOUND);
        const float_64 stepwidth(SYNC_FUNCS_BESSEL_INTEGRAL_STEPWIDTH);
        integrate(BesselK(), integral_result, x, upper_bound, stepwidth);

        return x * integral_result[0];
    }
    /** Second synchrotron function
     */
    float_64 F_2(const float_64 x) const
    {
        if(x == float_64(0.0))
            return float_64(0.0);

        return x * boost::math::tr1::cyl_bessel_k(2.0/3.0, x);
    }

public:
    enum Select
    {
        first=0, second=1
    };

    void init()
    {
        const uint32_t numSamples = SYNC_FUNCS_NUM_SAMPLES;

        this->dBuf_SyncFuncs[first] = MyBuf(new PMacc::container::DeviceBuffer<float_X, DIM1>(numSamples));
        this->dBuf_SyncFuncs[second] = MyBuf(new PMacc::container::DeviceBuffer<float_X, DIM1>(numSamples));

        PMacc::container::HostBuffer<float_X, DIM1> hBuf_F_1(numSamples);
        PMacc::container::HostBuffer<float_X, DIM1> hBuf_F_2(numSamples);

        for(uint32_t sampleIdx = 0u; sampleIdx < numSamples; sampleIdx++)
        {
            const float_64 x_m = float_64(sampleIdx) * SYNC_FUNCS_STEP_WIDTH;
            /* This mapping increases the sample point density for small values of x
             * where the synchrotron functions have a divergent slope. Without this mapping
             * the emission probabilty of low-energy photons is underestimated.
             */
            const float_64 x = x_m * x_m * x_m;

            hBuf_F_1.origin()[sampleIdx] = static_cast<float_X>(this->F_1(x));
            hBuf_F_2.origin()[sampleIdx] = static_cast<float_X>(this->F_2(x));
        }

        *this->dBuf_SyncFuncs[first] = hBuf_F_1;
        *this->dBuf_SyncFuncs[second] = hBuf_F_2;
    }

    /** Return a cursor representing a synchrotron function
     *
     * @param syncFunction first or second synchrotron function
     * @see: SynchrotronFunctions::Select
     */
    SyncFuncCursor getCursor(Select syncFunction) const
    {
        using namespace PMacc;

        detail::MapToLookupTable::LinInterpCursor linInterpCursor =
            cursor::tools::LinearInterp1D<float_X>()(this->dBuf_SyncFuncs[syncFunction]->origin());

        return cursor::make_Cursor(
            detail::MapToLookupTable(linInterpCursor),
            cursor::PlusNavigator(),
            float_X(0.0));
    }

}; // class SynchrotronFunctions

} // namespace bremsstrahlung
} // namespace particles
} // namespace picongpu
