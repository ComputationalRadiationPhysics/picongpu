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
#include "pmacc/cuSTL/cursor/accessor/CursorAccessor.hpp"

namespace pmacc
{
    namespace cursor
    {
        namespace tools
        {
            /** Returns a new cursor which looks like a rotated version of the one passed.
             *
             * The new cursor wraps the one that is passed. In the new cursor's navigator
             * the components of the passed int-vector are reordered according to the Axes
             * parameter and then passed to the nested cursor.
             *
             * \tparam Axes compile-time vector (pmacc::math::CT::Int) that descripes the mapping.
             * x-axis -> Axes::at<0>, y-axis -> Axes::at<1>, ...
             */
            template<typename Axes, typename TCursor>
            HDINLINE Cursor<CursorAccessor<TCursor>, CT::TwistAxesNavigator<Axes>, TCursor> twistAxes(
                const TCursor& cursor)
            {
                return Cursor<CursorAccessor<TCursor>, CT::TwistAxesNavigator<Axes>, TCursor>(
                    CursorAccessor<TCursor>(),
                    CT::TwistAxesNavigator<Axes>(),
                    cursor);
            }

        } // namespace tools
    } // namespace cursor
} // namespace pmacc
