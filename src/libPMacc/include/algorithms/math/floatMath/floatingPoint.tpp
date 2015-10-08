/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Richard Pausch,
 *                     Alexander Grund
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
#include "math.h"
#include <limits>

namespace PMacc
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
        /* Round towards nearest with x.5 rounded to +inf but take care of
         * floating point precision (e.g. 8388609.0 + 0.5 == 8388610.0) */
        float intPart;
        float fracPart = std::modf(value, &intPart);
        result res = float2int_rd(fracPart + 0.5f) + static_cast<int>(intPart);
        /* If we were close to x.5 then make sure res is even */
        if( abs(0.5f - (res - value)) < std::numeric_limits<float>::epsilon() )
            return res & ~1; /* Cancel out last bit of integer which makes it even */
        else
            return res;
#endif
    }
};

} //namespace math
} //namespace algorithms
} // namespace PMacc
