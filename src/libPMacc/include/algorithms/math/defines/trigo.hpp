/**
 * Copyright 2013 Heiko Burau, René Widera, Richard Pausch
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
 
/* 
 * File:   math.hpp
 * Author: widera
 *
 * Created on 30. Januar 2013, 09:55
 */

#pragma once

namespace PMacc
{
namespace algorithms
{

namespace math
{
namespace detail
{

template<typename Type>
struct Sin;

template<typename Type>
struct Cos;

template<typename ArgType, typename SinType, typename CosType>
struct SinCos;

template<typename Type>
struct Sinc;



} //namespace detail

template<typename T1>
HDINLINE static typename detail::Sin< T1 >::result sin(const T1& value)
{
    return detail::Sin< T1 > ()(value);
}

template<typename T1>
HDINLINE static typename detail::Cos<T1>::result cos(const T1& value)
{
    return detail::Cos< T1 > ()(value);
}

template<typename ArgType, typename SinType, typename CosType>
HDINLINE static typename detail::SinCos< ArgType, SinType, CosType >::result sincos(ArgType arg, SinType& sinValue, CosType& cosValue)
{
    return detail::SinCos< ArgType, SinType, CosType > ()(arg, sinValue, cosValue);
}

template<typename T1>
HDINLINE static typename detail::Sinc<T1>::result sinc(const T1& value)
{
    return detail::Sinc< T1 > ()(value);
}





} //namespace math
} //namespace algorithms
}//namespace PMacc

