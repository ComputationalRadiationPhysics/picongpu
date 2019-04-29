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
struct Abs<double>
{
    typedef double result;

    HDINLINE double operator( )(double value)
    {
#ifdef __CUDA_ARCH__
      return ::fabs( value );
#else
      /* \bug on cpu `::abs(double)` always return zero -> maybe this is the
       * integer version of `abs()`
       */
      return std::abs( value );
#endif
    }
};

template<>
struct Abs2<double>
{
    typedef double result;

    HDINLINE double operator( )(const double& value )
    {
        return value*value;
    }
};

} //namespace math
} //namespace algorithms
} // namespace pmacc
