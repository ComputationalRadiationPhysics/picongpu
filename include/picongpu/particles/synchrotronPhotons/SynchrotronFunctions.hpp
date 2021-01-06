/* Copyright 2015-2021 Heiko Burau
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

#include "picongpu/simulation_defines.hpp"

#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/cuSTL/container/DeviceBuffer.hpp>
#include <pmacc/cuSTL/cursor/Cursor.hpp>
#include <pmacc/cuSTL/cursor/navigator/PlusNavigator.hpp>
#include <pmacc/cuSTL/cursor/tools/LinearInterp.hpp>
#include <pmacc/cuSTL/cursor/BufferCursor.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/math/tr1.hpp> /* cyl_bessel_k */

namespace picongpu
{
    namespace particles
    {
        namespace synchrotronPhotons
        {
            namespace detail
            {
                /** Map `x` to the internal lookup table and return the result of the
                 * first or the second synchrotron function for `x`.
                 */
                struct MapToLookupTable
                {
                    using LinInterpCursor = typename ::pmacc::result_of::Functor<
                        ::pmacc::cursor::tools::LinearInterp<float_X>,
                        ::pmacc::cursor::BufferCursor<float_X, DIM1>>::type;

                    using type = float_X;

                    LinInterpCursor linInterpCursor;

                    /** constructor
                     *
                     * @param linInterpCursor lookup table of the first or the second
                     * synchrotron function.
                     */
                    HDINLINE MapToLookupTable(LinInterpCursor linInterpCursor) : linInterpCursor(linInterpCursor)
                    {
                    }

                    /** Returns F_1(x) or F_2(x)

                     * @param x position of the synchrotron function to be evaluated
                     */
                    HDINLINE float_X operator()(const float_X x) const;
                };

                using SyncFuncCursor
                    = ::pmacc::cursor::Cursor<MapToLookupTable, ::pmacc::cursor::PlusNavigator, float_X>;

            } // namespace detail


            /** Lookup table for synchrotron functions.
             *
             * Provides cursors for the first and the second synchrotron function
             */
            class SynchrotronFunctions
            {
            public:
                using SyncFuncCursor = detail::SyncFuncCursor;

            private:
                using MyBuf = boost::shared_ptr<pmacc::container::DeviceBuffer<float_X, DIM1>>;
                MyBuf dBuf_SyncFuncs[2]; // two synchrotron functions

                struct BesselK
                {
                    template<typename T_State, typename T_Time>
                    void operator()(const T_State& x, T_State& dxdt, T_Time t) const
                    {
                        dxdt[0] = boost::math::tr1::cyl_bessel_k(5.0 / 3.0, t);
                    }
                };

                /** First synchrotron function
                 */
                HINLINE float_64 F_1(const float_64 x) const;
                /** Second synchrotron function
                 */
                HINLINE float_64 F_2(const float_64 x) const;

            public:
                enum Select
                {
                    first = 0,
                    second = 1
                };

                HINLINE void init();
                /** Return a cursor representing a synchrotron function
                 *
                 * @param syncFunction first or second synchrotron function
                 * @see: SynchrotronFunctions::Select
                 */
                HINLINE SyncFuncCursor getCursor(Select syncFunction) const;

            }; // class SynchrotronFunctions

        } // namespace synchrotronPhotons
    } // namespace particles
} // namespace picongpu
