/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/static_assert.hpp"

#include <climits>
#include <type_traits>

namespace pmacc
{
    /**
     * Reverses the bit in an unsigned integral value
     *
     * Based on "Bit Twiddling Hacks" by Sean Eron Anderson
     * published in public domain. Retrieved on 13th of August, 2015 from
     * http://www.graphics.stanford.edu/~seander/bithacks.html
     *
     * @param value Value which should be reversed
     * @return Reversed value
     */
    template<typename T>
    T reverseBits(T value)
    {
        PMACC_STATIC_ASSERT_MSG(
            std::is_integral_v<T> && std::is_unsigned_v<T>,
            Only_allowed_for_unsigned_integral_types, );
        /* init with value (to get LSB) */
        T result = value;
        /* extra shift needed at end */
        int s = sizeof(T) * CHAR_BIT - 1;
        for(value >>= 1; value; value >>= 1)
        {
            result <<= 1;
            result |= value & 1;
            s--;
        }
        /* shift when values highest bits are zero */
        result <<= s;
        return result;
    }

} // namespace pmacc
