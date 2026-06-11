/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch, Axel Huebl, Alexander Debus
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
            using result = void;

            HDINLINE void operator()(float arg, float& sinValue, float& cosValue)
            {
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
                sinValue = pmacc::math::sin(arg);
                cosValue = pmacc::math::cos(arg);
#else
                ::sincosf(arg, &sinValue, &cosValue);
#endif
            }
        };

        template<>
        struct Sinc<float>
        {
            using result = float;

            HDINLINE float operator()(float const& value)
            {
                if(pmacc::math::abs(value) < FLT_EPSILON)
                    return 1.0f;
                else
                    return pmacc::math::sin(value) / value;
            }
        };

    } // namespace math
} // namespace pmacc
