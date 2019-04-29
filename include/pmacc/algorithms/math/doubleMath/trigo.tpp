/* Copyright 2013-2019 Heiko Burau, Rene Widera, Richard Pausch,
 *                     Axel Huebl, Alexander Debus
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

#include "pmacc/types.hpp"
#include <cfloat>
#include <cmath>


namespace pmacc
{
namespace algorithms
{
namespace math
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
struct ASin<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value)
    {
        return ::asin( value );
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
struct ACos<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value)
    {
        return ::acos( value );
    }
};

template<>
struct Tan<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value )
    {
        return ::tan( value );
    }
};

template<>
struct ATan<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value)
    {
        return ::atan( value );
    }
};

template<>
struct SinCos<double, double, double>
{
    typedef void result;

    HDINLINE void operator( )(double arg, double& sinValue, double& cosValue )
    {
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
        sinValue = ::sin(arg);
        cosValue = ::cos(arg);
#else
        ::sincos(arg, &sinValue, &cosValue);
#endif
    }
};


template<>
struct Sinc<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value )
    {
        if(pmacc::algorithms::math::abs(value) < DBL_EPSILON)
            return 1.0;
        else
            return pmacc::algorithms::math::sin( value )/value;
    }
};

template<>
struct Atan2<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& val1, const double& val2 )
    {
        return ::atan2( val1, val2 );
    }
};

} //namespace math
} //namespace algorithms
} // namespace pmacc
