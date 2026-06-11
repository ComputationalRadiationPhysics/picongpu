/*
 * SPDX-FileCopyrightText: Heiko Burau
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

#include <cmath>

namespace pmacc
{
    namespace math
    {
        template<>
        struct Modf<float>
        {
            using result = float;

            HDINLINE float operator()(float value, float* intpart)
            {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                return ::modff(value, intpart);
#else
                return std::modf(value, intpart);
#endif
            }
        };

    } // namespace math
} // namespace pmacc
