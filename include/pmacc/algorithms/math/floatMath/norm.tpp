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
        struct Norm<float>
        {
            using result = float;

            HDINLINE float operator()(float const& value)
            {
                return value * value;
            }
        };
    } // namespace math
} // namespace pmacc
