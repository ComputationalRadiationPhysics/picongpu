/* Copyright 2013-2019 Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Richard Pausch
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
struct Sqrt<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value )
    {
        return ::sqrt( value );
    }
};

template<>
struct RSqrt<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value )
    {
#if !defined(__CUDACC__)
        return 1.0/::sqrt(value);
#else
        return ::rsqrt(value);
#endif
    }
};

} //namespace math
} //namespace algorithms
} // namespace pmacc
