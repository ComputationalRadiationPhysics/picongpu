/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch,
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
    namespace math
    {
        template<>
        struct SinCos<float, float, float>
        {
            typedef void result;

            HDINLINE void operator()(float arg, float& sinValue, float& cosValue)
            {
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
                sinValue = cupla::math::sin(arg);
                cosValue = cupla::math::cos(arg);
#else
                ::sincosf(arg, &sinValue, &cosValue);
#endif
            }
        };

        template<>
        struct Sinc<float>
        {
            typedef float result;

            HDINLINE float operator()(const float& value)
            {
                if(cupla::math::abs(value) < FLT_EPSILON)
                    return 1.0f;
                else
                    return cupla::math::sin(value) / value;
            }
        };

    } // namespace math
} // namespace pmacc
