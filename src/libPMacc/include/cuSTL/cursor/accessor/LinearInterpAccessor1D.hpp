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

#include "types.h"
#include "algorithms/math/defines/modf.hpp"

namespace PMacc
{
namespace cursor
{

/** Performs a 1D, linear interpolation on access.
 *
 * \tparam T_Cursor 1D input data
 * \tparam T_Position type of the weighting factor
 */
template<typename T_Cursor, typename T_Position = float>
struct LinearInterpAccessor1D
{
    typedef typename T_Cursor::ValueType type;

    T_Cursor cursor;

    /**
     * @param cursor 1D input data
     */
    HDINLINE LinearInterpAccessor1D(const T_Cursor& cursor) : cursor(cursor) {}

    HDINLINE type operator()(const T_Position x) const
    {
        T_Position intPart;
        const T_Position fracPart = PMacc::algorithms::math::modf(x, &intPart);
        int idx = static_cast<int>(intPart);

        return (T_Position(1.0) - fracPart) * this->cursor[idx]
                                 + fracPart * this->cursor[idx+1];
    }
};

} // namespace cursor
} // namespace PMacc

