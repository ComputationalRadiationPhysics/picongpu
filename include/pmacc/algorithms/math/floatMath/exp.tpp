/* Copyright 2013-2019 Heiko Burau, Rene Widera, Richard Pausch
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


namespace pmacc
{
namespace algorithms
{
namespace math
{

template<>
struct Exp<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value )
    {
        return ::expf( value );
    }
};

template<>
struct Log<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value )
    {
        return ::logf( value );
    }
};

template<>
struct Log10<float>
{
    typedef float result;

    HDINLINE float operator( )(const float& value)
    {
#if __CUDA_ARCH__
        return ::log10f( value );
#else
        return ::log10( value );
#endif
    }
};

} //namespace math
} //namespace algorithms
} // namespace pmacc
