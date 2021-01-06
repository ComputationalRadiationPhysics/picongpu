/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
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
        struct Float : public Vector<float, dim>
        {
            using BaseType = Vector<float, dim>;

            HDINLINE Float()
            {
            }

            HDINLINE Float(float x) : BaseType(x)
            {
            }

            HDINLINE Float(float x, float y) : BaseType(x, y)
            {
            }

            HDINLINE Float(float x, float y, float z) : BaseType(x, y, z)
            {
            }

            /*! only allow explicit cast*/
            template<
                typename T_OtherType,
                typename T_OtherAccessor,
                typename T_OtherNavigator,
                template<typename, int>
                class T_OtherStorage>
            HDINLINE explicit Float(
                const Vector<T_OtherType, dim, T_OtherAccessor, T_OtherNavigator, T_OtherStorage>& vec)
                : BaseType(vec)
            {
            }

            HDINLINE Float(const BaseType& vec) : BaseType(vec)
            {
            }
        };

    } // namespace math
} // namespace pmacc
