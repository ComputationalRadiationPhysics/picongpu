/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cuSTL/cursor/Cursor.hpp>
#include <cuSTL/cursor/navigator/compile-time/TwistAxesNavigator.hpp>
#include <cuSTL/cursor/accessor/TwistAxesAccessor.hpp>

namespace PMacc
{
namespace cursor
{
namespace tools
{

namespace result_of
{

template<typename Axes, typename TCursor>
struct TwistVectorFieldAxes
{
    typedef Cursor<TwistAxesAccessor<TCursor, Axes>,
                   PMacc::cursor::CT::TwistAxesNavigator<Axes>,
                   TCursor> type;
};

} // result_of

/** Returns a new cursor which looks like a vector field rotated version of the one passed
 *
 * When rotating a vector field in physics the coordinate system and the vectors themselves
 * have to be rotated. This is the idea behind this function. It is assuming that the cursor
 * which is passed returns in its access call a vector type of the same dimension as in
 * the jumping call. In other words, the field and the vector have the same dimension.
 *
 * e.g.: new_cur = twistVectorFieldAxes<math::CT::Int<1,2,0> >(cur); // x -> y, y -> z, z -> x
 *
 * \tparam Axes compile-time vector (PMacc::math::CT::Int) that describes the mapping.
 * x-axis -> Axes::at<0>, y-axis -> Axes::at<1>, ...
 *
 */
template<typename Axes, typename TCursor>
HDINLINE
typename result_of::TwistVectorFieldAxes<Axes, TCursor>::type
twistVectorFieldAxes(const TCursor& cursor)
{
    return typename result_of::TwistVectorFieldAxes<Axes, TCursor>::type
        (TwistAxesAccessor<TCursor, Axes>(),
        PMacc::cursor::CT::TwistAxesNavigator<Axes>(),
        cursor);
}

} // tools
} // cursor
} // PMacc
