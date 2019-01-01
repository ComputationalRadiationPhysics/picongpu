/* Copyright 2013-2019 Heiko Burau, Rene Widera, Richard Pausch, Axel Huebl, Alexander Debus
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
struct Sin;

template<typename Type>
struct ASin;

template<typename Type>
struct Cos;

template<typename Type>
struct ACos;

template<typename Type>
struct Tan;

template<typename Type>
struct ATan;

template<typename Type>
struct Atan2;

template<typename ArgType, typename SinType, typename CosType>
struct SinCos;

template<typename Type>
struct Sinc;


template<typename T1>
HDINLINE
typename Sin< T1 >::result
sin(const T1& value)
{
    return Sin< T1 > ()(value);
}

template<typename T1>
HDINLINE
typename ASin< T1 >::result
asin(const T1& value)
{
    return ASin< T1 > ()(value);
}

template<typename T1>
HDINLINE
typename Cos<T1>::result
cos(const T1& value)
{
    return Cos< T1 > ()(value);
}

template<typename T1>
HDINLINE
typename ACos<T1>::result
acos(const T1& value)
{
    return ACos< T1 > ()(value);
}

template<typename T1>
HDINLINE
typename Tan<T1>::result
tan(const T1& value)
{
    return Tan< T1 > ()(value);
}

template<typename T1>
HDINLINE
typename ATan<T1>::result
atan(const T1& value)
{
    return ATan< T1 > ()(value);
}

template<typename ArgType, typename SinType, typename CosType>
HDINLINE
typename SinCos< ArgType, SinType, CosType >::result
sincos(ArgType arg, SinType& sinValue, CosType& cosValue)
{
    return SinCos< ArgType, SinType, CosType > ()(arg, sinValue, cosValue);
}

template<typename T1>
HDINLINE
typename Sinc<T1>::result
sinc(const T1& value)
{
    return Sinc< T1 > ()(value);
}

template<typename T1>
HDINLINE
typename Atan2<T1>::result
atan2(const T1& val1, const T1& val2)
{
    return Atan2< T1 > ()(val1, val2);
}

} /* namespace math */
} /* namespace algorithms */
} /* namespace pmacc */
