/**
 * Copyright 2013-2014 Rene Widera
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
#include <math.h> /*provide host version of pow*/

namespace PMacc
{
namespace algorithms
{
namespace math
{

/*C++98 standard define a separate version for int and float exponent*/

template<>
struct Pow<float, float>
{
    typedef float result;

    HDINLINE result operator()(const float& base, const float& exponent)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::powf(base, exponent);
#else
        return ::pow(base, exponent);
#endif

    }
};

template<>
struct Pow<float, int>
{
    typedef float result;

    HDINLINE result operator()(const float& base,const int& exponent)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        return ::powf(base, exponent);
#else
        return ::pow(base, exponent);
#endif

    }
};

} //namespace math
} //namespace algorithms
} // namespace PMacc
