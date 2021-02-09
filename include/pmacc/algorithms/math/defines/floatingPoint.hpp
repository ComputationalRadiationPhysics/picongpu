/* Copyright 2013-2021 Heiko Burau, Rene Widera, Alexander Grund
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

namespace pmacc
{
    namespace math
    {
        template<typename Type>
        struct Float2int_ru;

        template<typename Type>
        struct Float2int_rd;

        template<typename Type>
        struct Float2int_rn;

        /**
         * Returns the smallest int value that is at least as big as value
         * Note: Using values outside the range of an int is undefined
         * @return integer value
         */
        template<typename T1>
        HDINLINE typename Float2int_ru<T1>::result float2int_ru(T1 value)
        {
            return Float2int_ru<T1>()(value);
        }

        /**
         * Returns the largest int value that is not greater than value
         * Note: Using values outside the range of an int is undefined
         * @return integer value
         */
        template<typename T1>
        HDINLINE typename Float2int_rd<T1>::result float2int_rd(T1 value)
        {
            return Float2int_rd<T1>()(value);
        }

        /**
         * Rounds towards the nearest value returning an int
         * For the case of x.5 the even value is chosen from the 2 possible values
         * Note: Using values outside the range of an int is undefined
         * @return integer value
         */
        template<typename T1>
        HDINLINE typename Float2int_rn<T1>::result float2int_rn(T1 value)
        {
            return Float2int_rn<T1>()(value);
        }

    } // namespace math
} // namespace pmacc
