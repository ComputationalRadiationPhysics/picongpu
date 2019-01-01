/* Copyright 2016-2019 Alexander Debus
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

namespace pmacc
{
namespace algorithms
{
namespace math
{

template<typename Type>
struct Fmod;

/**
 * Equivalent to the modulus-operator for float types
 * returns the floating-point remainder of x / y.
 * The functionality corresponds to the C++
 * math function fmod().
 * For details, see http://www.cplusplus.com/reference/cmath/fmod/ .
 * @return float value
 */
template<typename T1>
HDINLINE typename Fmod< T1>::result fmod(T1 x, T1 y)
{
    return Fmod< T1 > ()(x, y);
}

} //namespace math
} //namespace algorithms
} //namespace pmacc

