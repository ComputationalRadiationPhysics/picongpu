/* Copyright 2015-2019 Benjamin Worpitz, Richard Pausch
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
struct Min<float, float>
{
    typedef float result;

    HDINLINE float operator()(float value1, float value2)
    {
        return ::fminf(value1, value2);
    }
};

template<>
struct Max<float, float>
{
    typedef float result;

    HDINLINE float operator()(float value1, float value2)
    {
        return ::fmaxf(value1, value2);
    }
};

} //namespace math
} //namespace algorithms
} //namespace pmacc
