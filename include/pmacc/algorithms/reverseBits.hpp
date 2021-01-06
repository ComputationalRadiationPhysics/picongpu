/* Copyright 2015-2021 Alexander Grund
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

#include "pmacc/static_assert.hpp"
#include <boost/type_traits.hpp>
#include <climits>

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
            boost::is_integral<T>::value && boost::is_unsigned<T>::value,
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
