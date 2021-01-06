/* Copyright 2013-2021 Heiko Burau, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/cuSTL/cursor/Cursor.hpp"
#include "pmacc/cuSTL/cursor/navigator/compile-time/TwistAxesNavigator.hpp"
#include "pmacc/cuSTL/cursor/accessor/TwistAxesAccessor.hpp"

namespace pmacc
{
    namespace cursor
    {
        namespace tools
        {
            namespace result_of
            {
                /** result for TwistVectorFieldAxes
                 *
                 * \tparam T_NavigatorPerm permutation vector for navigator
                 * \tparam T_AccessorPerm permutation vector for the accessor
                 * \tparam T_Cursor cursor to permute
                 */
                template<typename T_NavigatorPerm, typename T_AccessorPerm, typename T_Cursor>
                struct TwistVectorFieldAxes
                {
                    typedef Cursor<
                        TwistAxesAccessor<T_Cursor, T_AccessorPerm>,
                        pmacc::cursor::CT::TwistAxesNavigator<T_NavigatorPerm>,
                        T_Cursor>
                        type;
                };

            } // namespace result_of

            /** Returns a new cursor which looks like a vector field rotated version of the one passed
             *
             * When rotating a vector field in physics the coordinate system and the vectors themselves
             * have to be rotated. This is the idea behind this function. It is assuming that the cursor
             * which is passed returns in its access call a vector type of the same dimension as in
             * the jumping call. In other words, the field and the vector have the same dimension.
             *
             * e.g.: new_cur = twistVectorFieldAxes<math::CT::Int<1,2,0> >(cur); // x -> y, y -> z, z -> x
             *
             * \tparam T_Permutation compile-time vector (pmacc::math::CT::Int) that describes the mapping.
             * x-axis -> T_Permutation::at<0>, y-axis -> T_Permutation::at<1>, ...
             *
             */
            template<typename T_Permutation, typename T_Cursor>
            HDINLINE typename result_of::TwistVectorFieldAxes<T_Permutation, T_Permutation, T_Cursor>::type
            twistVectorFieldAxes(const T_Cursor& cursor)
            {
                return typename result_of::TwistVectorFieldAxes<T_Permutation, T_Permutation, T_Cursor>::type(
                    TwistAxesAccessor<T_Cursor, T_Permutation>(),
                    pmacc::cursor::CT::TwistAxesNavigator<T_Permutation>(),
                    cursor);
            }

            /** permute navigation and access of a cursor
             *
             * use same permutation for accessor and navigator
             *
             * \tparam T_Permutation permutation vector
             * \tparam T_Cursor permutation vector
             * \param cursor cursor to permute
             * \param permutation cursor to permute
             */
            template<typename T_Cursor, typename T_Permutation>
            HDINLINE typename result_of::TwistVectorFieldAxes<T_Permutation, T_Permutation, T_Cursor>::type
            twistVectorFieldAxes(const T_Cursor& cursor, const T_Permutation& /*permutation*/)
            {
                return typename result_of::TwistVectorFieldAxes<T_Permutation, T_Permutation, T_Cursor>::type(
                    TwistAxesAccessor<T_Cursor, T_Permutation>(),
                    pmacc::cursor::CT::TwistAxesNavigator<T_Permutation>(),
                    cursor);
            }

            /** permute navigation and access of a cursor
             *
             * different dimensions for the accessor and navigator permutation vector are allowed
             *
             * \param cursor cursor to permute
             * \param navigatorPermutation compile time permutation vector for the navigator
             * \param accessorPermutation compile time permutation vector for the accessor
             */
            template<typename T_Cursor, typename T_NavigatorPerm, typename T_AccessorPerm>
            HDINLINE typename result_of::TwistVectorFieldAxes<T_NavigatorPerm, T_AccessorPerm, T_Cursor>::type
            twistVectorFieldAxes(
                const T_Cursor& cursor,
                const T_NavigatorPerm& /*navigatorPermutation*/,
                const T_AccessorPerm& /*accessorPermutation*/)
            {
                return typename result_of::TwistVectorFieldAxes<T_NavigatorPerm, T_AccessorPerm, T_Cursor>::type(
                    TwistAxesAccessor<T_Cursor, T_AccessorPerm>(),
                    pmacc::cursor::CT::TwistAxesNavigator<T_NavigatorPerm>(),
                    cursor);
            }

        } // namespace tools
    } // namespace cursor
} // namespace pmacc
