/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch, Axel Huebl, Alexander Debus
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

            HDINLINE double operator()(double const& value)
            {
                if(pmacc::math::abs(value) < DBL_EPSILON)
                    return 1.0;
                else
                    return pmacc::math::sin(value) / value;
            }
        };

    } // namespace math
} // namespace pmacc
