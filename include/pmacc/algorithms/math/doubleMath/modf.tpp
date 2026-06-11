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
        struct Modf<double>
        {
            using result = double;

            HDINLINE double operator()(double value, double* intpart)
            {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                return ::modf(value, intpart);
#else
                return std::modf(value, intpart);
#endif
            }
        };

    } // namespace math
} // namespace pmacc
