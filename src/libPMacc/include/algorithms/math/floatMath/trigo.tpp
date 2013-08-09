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
 * File:   abs.hpp
 * Author: widera
 *
 * Created on 30. Januar 2013, 09:50
 */

#pragma once

#include "types.h"
#include <float.h>

namespace PMacc
{
namespace algorithms
{
namespace math
{
namespace detail
{

template<>
struct Sin<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value )
    {
        return ::sinf( value );
    }
};

template<>
struct Cos<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value )
    {
        return ::cosf( value );
    }
};

template<>
struct SinCos<float, float, float>
{
    typedef void result;

    HDINLINE void operator( )(float arg, float& sinValue, float& cosValue )
    {
        ::sincosf( arg, &sinValue, &cosValue );
    }
};



template<>
struct Sinc<float>
{
    typedef float result;

    HDINLINE double operator( )(const float& value )
    {
      if(::fabsf(value) < FLT_EPSILON) 
	return 1.0;
      else 
	return ::sinf( value )/value;
    }
};






} //namespace detail
} //namespace math
} //namespace algorithms
} // namespace PMacc

