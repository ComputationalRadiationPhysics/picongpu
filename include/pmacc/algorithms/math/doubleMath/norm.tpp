/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch
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
        struct Norm<double>
        {
            using result = double;

            HDINLINE double operator()(double const& value)
            {
                return value * value;
            }
        };
    } // namespace math
} // namespace pmacc
