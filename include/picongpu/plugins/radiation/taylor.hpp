/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "VectorTypes.hpp"

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            struct Taylor
            {
                // a Taylor development for 1-sqrt(1-x)

                HDINLINE picongpu::float_64 operator()(picongpu::float_64 x) const
                {
                    // Taylor series of 1-sqrt(1-x) till 5th order
                    // same like 0.5*x + 0.125*x*x + 0.0625 * x*x*x + 0.0390625 * x*x*x*x + 0.02734375 *x*x*x*x*x;
                    picongpu::float_64 const x2 = (x * x);
                    return x * ((0.5 + 0.125 * x) + x2 * (0.0625 + (0.0390625 * x + 0.02734375 * x2)));
                }
            };

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
