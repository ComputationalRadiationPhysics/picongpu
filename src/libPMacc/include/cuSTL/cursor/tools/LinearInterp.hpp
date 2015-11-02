/**
 * Copyright 2015 Heiko Burau
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

#include "cuSTL/cursor/Cursor.hpp"
#include "cuSTL/cursor/accessor/LinearInterpAccessor.hpp"
#include "cuSTL/cursor/navigator/AbstractNavigator.hpp"
#include "types.h"

namespace PMacc
{
namespace cursor
{
namespace tools
{

/** Return a cursor that does 1D, linear interpolation on input data.
 *
 * \tparam T_Position type of the weighting factor
 *
 */
template<typename T_Position = float>
struct LinearInterp
{
    template<typename TCursor>
    HDINLINE
    Cursor<LinearInterpAccessor<TCursor, T_Position>, AbstractNavigator, T_Position>
    operator()(const TCursor& cur)
    {
        return make_Cursor(
            LinearInterpAccessor<TCursor, T_Position>(cur),
            AbstractNavigator(),
            T_Position(0.0));
    }
};

} // namespace tools
} // namespace cursor
} // namespace PMacc

