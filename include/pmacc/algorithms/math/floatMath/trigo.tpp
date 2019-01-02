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
struct Sin<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value )
    {
        return ::sinf( value );
    }
};

template<>
struct ASin<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value)
    {
#if __CUDA_ARCH__
        return ::asinf( value );
#else
        return ::asin( value );
#endif
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
struct ACos<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value)
    {
#if __CUDA_ARCH__
        return ::acosf( value );
#else
        return ::acos( value );
#endif
    }
};

template<>
struct Tan<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value )
    {
        return ::tanf( value );
    }
};

template<>
struct ATan<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value)
    {
        return ::atanf( value );
    }
};

template<>
struct SinCos<float, float, float>
{
    typedef void result;

    HDINLINE void operator( )(float arg, float& sinValue, float& cosValue )
    {
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
        sinValue = ::sinf(arg);
        cosValue = ::cosf(arg);
#else
        ::sincosf( arg, &sinValue, &cosValue );
#endif
    }
};



template<>
struct Sinc<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value )
    {
        if(pmacc::algorithms::math::abs(value) < FLT_EPSILON)
            return 1.0;
        else
            return pmacc::algorithms::math::sin( value )/value;
    }
};

template<>
struct Atan2<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& val1, const float& val2 )
    {
        return ::atan2f( val1, val2 );
    }
};

} //namespace math
} //namespace algorithms
} // namespace pmacc
