/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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

#include <pmacc/types.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <pmacc/result_of_Functor.hpp>


// forward declaration
namespace picongpu
{
    struct AssignedTrilinearInterpolation;
} // namespace picongpu

namespace pmacc
{
    namespace result_of
    {
        template<typename T_Cursor>
        struct Functor<picongpu::AssignedTrilinearInterpolation, T_Cursor>
        {
            using type = typename boost::remove_reference<typename T_Cursor::type>::type;
        };

    } // namespace result_of
} // namespace pmacc

namespace picongpu
{
    struct AssignedTrilinearInterpolation
    {
        /** Does a 3D trilinear field-to-point interpolation for
         * arbitrary assignment function and arbitrary field_value types.
         *
         * \tparam T_AssignmentFunction function for assignment
         * \tparam T_begin lower margin for interpolation
         * \tparam T_end upper margin for interpolation
         *
         * \param cursor cursor pointing to the field
         * \param pos position of the interpolation point
         * \return sum over: field_value * assignment
         *
         * interpolate on grid points in range [T_begin;T_end]
         */
        template<typename T_AssignmentFunction, int T_begin, int T_end, typename T_Cursor>
        HDINLINE static auto interpolate(const T_Cursor& cursor, const float3_X& pos) ->
            typename ::pmacc::result_of::Functor<AssignedTrilinearInterpolation, T_Cursor>::type
        {
            using type = typename ::pmacc::result_of::Functor<AssignedTrilinearInterpolation, T_Cursor>::type;

            type result_z = type(0.0);
            for(int z = T_begin; z <= T_end; ++z)
            {
                type result_y = type(0.0);
                for(int y = T_begin; y <= T_end; ++y)
                {
                    type result_x = type(0.0);
                    for(int x = T_begin; x <= T_end; ++x)
                        /* a form factor is the "amount of particle" that is affected by this cell
                         * so we have to sum over: cell_value * form_factor
                         */
                        result_x += *cursor(x, y, z) * T_AssignmentFunction()(float_X(x) - pos.x());

                    result_y += result_x * T_AssignmentFunction()(float_X(y) - pos.y());
                }

                result_z += result_y * T_AssignmentFunction()(float_X(z) - pos.z());
            }
            return result_z;
        }

        /** Implementation for 2D position*/
        template<class T_AssignmentFunction, int T_begin, int T_end, class T_Cursor>
        HDINLINE static auto interpolate(T_Cursor const& cursor, float2_X const& pos) ->
            typename ::pmacc::result_of::Functor<AssignedTrilinearInterpolation, T_Cursor>::type
        {
            using type = typename ::pmacc::result_of::Functor<AssignedTrilinearInterpolation, T_Cursor>::type;

            type result_y = type(0.0);
            for(int y = T_begin; y <= T_end; ++y)
            {
                type result_x = type(0.0);
                for(int x = T_begin; x <= T_end; ++x)
                    // a form factor is the "amount of particle" that is affected by this cell
                    // so we have to sum over: cell_value * form_factor
                    result_x += *cursor(x, y) * T_AssignmentFunction()(float_X(x) - pos.x());

                result_y += result_x * T_AssignmentFunction()(float_X(y) - pos.y());
            }
            return result_y;
        }

        static auto getStringProperties() -> pmacc::traits::StringProperty
        {
            pmacc::traits::StringProperty propList("name", "uniform");
            return propList;
        }
    };

} // namespace picongpu
