/**
 * Copyright 2013-2016 Rene Widera, Alexander Grund
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
#include <cmath>

namespace PMacc
{
namespace algorithms
{
namespace math
{
/*C++98 standard define a separate version for int and double exponent*/

template<>
struct Pow<double, double>
{
    typedef double result;

    HDINLINE result operator()(const double& base, const double& exponent)
    {
        return ::pow(base, exponent);
    }
};

template<>
struct Pow<double, int>
{
    typedef double result;

    HDINLINE result operator()(const double& base, const int& exponent)
    {
#ifdef __CUDA_ARCH__ /*device version*/
        /* @todo: There is an incompatibility with C++11 + CUDA + GCC where no device function
         *        pow(double, int) is defined: http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-tools-title-known
         *        Use the pow(double, double) instead which reduces performance or implement an own (faster) version
         */
        return ::pow(base, static_cast<double>(exponent));
#else
        return ::pow(base, exponent);
#endif
    }
};


} //namespace math
} //namespace algorithms
} // namespace PMacc
