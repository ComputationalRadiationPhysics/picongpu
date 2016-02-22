/**
 * Copyright 2015-2016 Heiko Burau
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
#include "cuSTL/cursor/accessor/LinearInterpAccessor1D.hpp"
#include "cuSTL/cursor/navigator/PlusNavigator.hpp"
#include "result_of_Functor.hpp"
#include "pmacc_types.hpp"

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
struct LinearInterp1D
{
    template<typename TCursor>
    HDINLINE
    Cursor<LinearInterpAccessor1D<TCursor, T_Position>, PlusNavigator, T_Position>
    operator()(const TCursor& cur)
    {
        return make_Cursor(
            LinearInterpAccessor1D<TCursor, T_Position>(cur),
            PlusNavigator(),
            T_Position(0.0));
    }
};

} // namespace tools
} // namespace cursor

namespace result_of
{

template<typename T_Position, typename TCursor>
struct Functor<cursor::tools::LinearInterp1D<T_Position>, TCursor>
{
    typedef cursor::Cursor<
        cursor::LinearInterpAccessor1D<TCursor, T_Position>,
        cursor::PlusNavigator,
        T_Position> type;
};

} // namespace result_of

} // namespace PMacc

