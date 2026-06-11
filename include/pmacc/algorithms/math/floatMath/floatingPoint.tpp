/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/types.hpp"

#include <cmath>
#include <limits>

namespace pmacc
{
    namespace math
    {
        template<>
        struct Float2int_ru<float>
        {
            using result = int;

            HDINLINE result operator()(float value)
            {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                return ::__float2int_ru(value);
#else
                return static_cast<int>(ceil(value));
#endif
            }
        };

        template<>
        struct Float2int_rd<float>
        {
            using result = int;

            HDINLINE result operator()(float value)
            {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                return ::__float2int_rd(value);
#else
                return static_cast<int>(floor(value));
#endif
            }
        };

        template<>
        struct Float2int_rn<float>
        {
            using result = int;

            HDINLINE result operator()(float value)
            {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                return ::__float2int_rn(value);
#else
                if(value < 0.0f)
                    return -(*this)(-value);
                float intPart;
                float fracPart = std::modf(value, &intPart);
                auto res = static_cast<int>(intPart);
                /* epsilon in the following code is used to consider values
                 * "very close" to x.5 also as x.5
                 */
                if(fracPart > 0.5f + std::numeric_limits<float>::epsilon())
                {
                    /* >x.5 --> Round up */
                    res = res + 1;
                }
                else if(!(fracPart < 0.5f - std::numeric_limits<float>::epsilon()))
                {
                    /* We are NOT >x.5 AND NOT <x.5 --> ==x.5 --> use x if x is even, else x+1
                     * The "&~1" cancels the last bit which results in an even value
                     * res is even -> res+1 is odd -> (res+1)&~1 = res
                     * res is odd -> res+1 is even -> (res+1)&~1 = res+1
                     */
                    res = (res + 1) & ~1;
                }
                /* else res = res (round down) */
                return res;
#endif
            }
        };

    } // namespace math
} // namespace pmacc
