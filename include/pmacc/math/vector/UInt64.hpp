/* Copyright 2013-2021 Heiko Burau, Rene Widera, Axel Huebl
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

#include "Vector.hpp"

namespace pmacc
{
    namespace math
    {
        template<int dim>
        struct UInt64 : public Vector<uint64_t, dim>
        {
            using BaseType = Vector<uint64_t, dim>;

            HDINLINE UInt64()
            {
            }

            HDINLINE UInt64(uint64_t x) : BaseType(x)
            {
            }

            HDINLINE UInt64(uint64_t x, uint64_t y) : BaseType(x, y)
            {
            }

            HDINLINE UInt64(uint64_t x, uint64_t y, uint64_t z) : BaseType(x, y, z)
            {
            }

            /*! only allow explicit cast*/
            template<
                typename T_OtherType,
                typename T_OtherAccessor,
                typename T_OtherNavigator,
                template<typename, int>
                class T_OtherStorage>
            HDINLINE explicit UInt64(
                const Vector<T_OtherType, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& vec)
                : BaseType(vec)
            {
            }

            HDINLINE UInt64(const BaseType& vec) : BaseType(vec)
            {
            }
        };

    } // namespace math
} // namespace pmacc
