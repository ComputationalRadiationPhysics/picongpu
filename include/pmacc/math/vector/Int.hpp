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

#include "Vector.hpp"

namespace pmacc
{
    namespace math
    {
        template<int dim>
        struct Int : public Vector<int, dim>
        {
            using BaseType = Vector<int, dim>;

            HDINLINE Int()
            {
            }

            HDINLINE Int(int x) : BaseType(x)
            {
            }

            HDINLINE Int(int x, int y) : BaseType(x, y)
            {
            }

            HDINLINE Int(int x, int y, int z) : BaseType(x, y, z)
            {
            }

            /*! only allow explicit cast*/
            template<
                typename T_OtherType,
                typename T_OtherAccessor,
                typename T_OtherNavigator,
                template<typename, int>
                class T_OtherStorage>
            HDINLINE explicit Int(
                const Vector<T_OtherType, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& vec)
                : BaseType(vec)
            {
            }

            HDINLINE Int(const BaseType& vec) : BaseType(vec)
            {
            }
        };

    } // namespace math
} // namespace pmacc
