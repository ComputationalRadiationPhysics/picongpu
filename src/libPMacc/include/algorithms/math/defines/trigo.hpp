/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera, Richard Pausch
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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

namespace PMacc
{
namespace algorithms
{

namespace math
{

template<typename Type>
struct Sin;

template<typename Type>
struct Cos;

template<typename ArgType, typename SinType, typename CosType>
struct SinCos;

template<typename Type>
struct Sinc;


template<typename T1>
HDINLINE static typename Sin< T1 >::result sin(const T1& value)
{
    return Sin< T1 > ()(value);
}

template<typename T1>
HDINLINE static typename Cos<T1>::result cos(const T1& value)
{
    return Cos< T1 > ()(value);
}

template<typename ArgType, typename SinType, typename CosType>
HDINLINE static typename SinCos< ArgType, SinType, CosType >::result sincos(ArgType arg, SinType& sinValue, CosType& cosValue)
{
    return SinCos< ArgType, SinType, CosType > ()(arg, sinValue, cosValue);
}

template<typename T1>
HDINLINE static typename Sinc<T1>::result sinc(const T1& value)
{
    return Sinc< T1 > ()(value);
}

} //namespace math
} //namespace algorithms
}//namespace PMacc
