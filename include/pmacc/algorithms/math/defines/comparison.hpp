/* Copyright 2013-2021 Heiko Burau, Rene Widera
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
        template<typename T1, typename T2>
        struct Max;

        template<typename T1, typename T2>
        struct Min;

        template<typename T>
        struct Max<T, T>
        {
            typedef T result;

            HDINLINE T operator()(T value1, T value2)
            {
                return value1 > value2 ? value1 : value2;
            }
        };

        template<typename T>
        struct Min<T, T>
        {
            typedef T result;

            HDINLINE T operator()(T value1, T value2)
            {
                return value1 < value2 ? value1 : value2;
            }
        };

        template<typename T1, typename T2>
        HDINLINE typename Min<T1, T2>::result min(const T1& value1, const T2& value2)
        {
            return Min<T1, T2>()(value1, value2);
        }

        template<typename T1, typename T2>
        HDINLINE typename Max<T1, T2>::result max(const T1& value1, const T2& value2)
        {
            return Max<T1, T2>()(value1, value2);
        }

    } // namespace math
} // namespace pmacc
