/* Copyright 2013-2023 Heiko Burau, Rene Widera, Richard Pausch,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include "pmacc/math/math.hpp"
#include "pmacc/types.hpp"

#include <cfloat>
#include <cmath>


namespace pmacc
{
    namespace math
    {
        template<>
        struct SinCos<double, double, double>
        {
            using result = void;

            HDINLINE void operator()(double arg, double& sinValue, double& cosValue)
            {
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
                sinValue = pmacc::math::sin(arg);
                cosValue = pmacc::math::cos(arg);
#else
                ::sincos(arg, &sinValue, &cosValue);
#endif
            }
        };


        template<>
        struct Sinc<double>
        {
            using result = double;

            HDINLINE double operator()(const double& value)
            {
                if(pmacc::math::abs(value) < DBL_EPSILON)
                    return 1.0;
                else
                    return pmacc::math::sin(value) / value;
            }
        };

    } // namespace math
} // namespace pmacc
