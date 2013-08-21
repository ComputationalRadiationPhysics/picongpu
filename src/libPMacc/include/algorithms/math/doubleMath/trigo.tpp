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
struct Sin<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value )
    {
        return ::sin( value );
    }
};

template<>
struct Cos<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value )
    {
        return ::cos( value );
    }
};

template<>
struct SinCos<double, double, double>
{
    typedef void result;

    HDINLINE void operator( )(double arg, double& sinValue, double& cosValue )
    {
        ::sincos( arg, &sinValue, &cosValue );
    }
};


template<>
struct Sinc<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value )
    {
      if(::fabs(value) < DBL_EPSILON) 
	return 1.0;
      else 
	return ::sin( value )/value;
    }
};





} //namespace detail
} //namespace math
} //namespace algorithms
} // namespace PMacc

