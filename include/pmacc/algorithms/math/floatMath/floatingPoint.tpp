/* Copyright 2013-2019 Heiko Burau, Rene Widera, Richard Pausch,
 *                     Alexander Grund
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
#include <cmath>
#include <limits>

namespace pmacc
{
namespace algorithms
{
namespace math
{

template<>
struct Floor<float>
{
    typedef float result;

    HDINLINE result operator( )(result value)
    {
        return ::floorf( value );
    }
};

template<>
struct Ceil<float>
{
    typedef float result;

    HDINLINE result operator( )(result value)
    {
        return ::ceil( value );
    }
};

template<>
struct Float2int_ru<float>
{
    typedef int result;

    HDINLINE result operator( )(float value)
    {
#if __CUDA_ARCH__
        return ::__float2int_ru( value );
#else
        return static_cast<int>(ceil(value));
#endif
    }
};

template<>
struct Float2int_rd<float>
{
    typedef int result;

    HDINLINE result operator( )(float value)
    {
#if __CUDA_ARCH__
        return ::__float2int_rd( value );
#else
        return static_cast<int>(floor(value));
#endif
    }
};

template<>
struct Float2int_rn<float>
{
    typedef int result;

    HDINLINE result operator( )(float value)
    {
#if __CUDA_ARCH__
        return ::__float2int_rn( value );
#else
        if(value < 0.0f)
            return -(*this)(-value);
        float intPart;
        float fracPart = std::modf(value, &intPart);
        result res = static_cast<int>(intPart);
        /* epsilon in the following code is used to consider values
         * "very close" to x.5 also as x.5
         */
        if(fracPart > 0.5f + std::numeric_limits<float>::epsilon())
        {
            /* >x.5 --> Round up */
            res = res + 1;
        }
        else if(!(fracPart < 0.5f - std::numeric_limits<float>::epsilon()))
        {
            /* We are NOT >x.5 AND NOT <x.5 --> ==x.5 --> use x if x is even, else x+1
             * The "&~1" cancels the last bit which results in an even value
             * res is even -> res+1 is odd -> (res+1)&~1 = res
             * res is odd -> res+1 is even -> (res+1)&~1 = res+1
             */
            res = (res + 1) & ~1;
        }
        /* else res = res (round down) */
        return res;
#endif
    }
};

} //namespace math
} //namespace algorithms
} // namespace pmacc
