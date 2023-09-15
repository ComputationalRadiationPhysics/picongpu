/* Copyright 2013-2023 Heiko Burau, Rene Widera
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/math/vector/Vector.hpp"
#include "pmacc/math/vector/navigator/PermutedNavigator.hpp"
#include "pmacc/math/vector/navigator/StackedNavigator.hpp"

namespace pmacc
{
    namespace math
    {
        namespace detail
        {
            template<typename T_Axes, typename T_Vector>
            struct TwistComponents;

            template<
                typename T_Axes,
                typename T_Type,
                uint32_t T_dim,
                typename T_Accessor,
                typename T_Navigator,
                typename T_Storage>
            struct TwistComponents<T_Axes, math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>>
            {
                using type = math::Vector<
                    T_Type,
                    T_dim,
                    T_Accessor,
                    math::StackedNavigator<T_Navigator, math::PermutedNavigator<T_Axes>>,
                    T_Storage>&;
            };

            template<typename T_Axes, unsigned dim>
            struct TwistComponents<T_Axes, DataSpace<dim>>
            {
                using type = typename TwistComponents<T_Axes, typename DataSpace<dim>::BaseType>::type;
            };

            template<
                typename T_Axes,
                typename T_Type,
                uint32_t T_dim,
                typename T_Accessor,
                typename T_Navigator,
                typename T_Storage>
            struct TwistComponents<T_Axes, const math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage>>
            {
                using type = math::Vector<
                    T_Type,
                    T_dim,
                    T_Accessor,
                    math::StackedNavigator<T_Navigator, math::PermutedNavigator<T_Axes>>,
                    T_Storage> const&;
            };

            template<typename T_Axes, unsigned dim>
            struct TwistComponents<T_Axes, const DataSpace<dim>>
            {
                using type = typename TwistComponents<T_Axes, const typename DataSpace<dim>::BaseType>::type;
            };

        } // namespace detail

        /** Returns a reference of vector with twisted axes.
         *
         * The axes twist operation is done in place. This means that the result refers to the
         * memory of the input vector. The input vector's navigator policy is replaced by
         * a new navigator which merely consists of the old navigator plus a twisting navigator.
         * This new navigator does not use any real memory.
         *
         * @tparam T_Axes Mapped indices
         * @tparam T_Vector type of vector to be twisted
         * @param vector vector to be twisted
         * @return reference of the input vector with twisted axes.
         */
        template<typename T_Axes, typename T_Vector>
        HDINLINE auto twistComponents(T_Vector& vector)
        {
            /* The reinterpret_cast is valid because the target type is the same as the
             * input type except its navigator policy which does not occupy any memory though.
             */
            return reinterpret_cast<typename detail::TwistComponents<T_Axes, T_Vector>::type>(vector);
        }
        template<typename T_Axes, typename T_Vector>
        HDINLINE auto twistComponents(T_Vector const& vector)
        {
            /* The reinterpret_cast is valid because the target type is the same as the
             * input type except its navigator policy which does not occupy any memory though.
             */
            return reinterpret_cast<typename detail::TwistComponents<T_Axes, T_Vector const>::type>(vector);
        }

    } // namespace math
} // namespace pmacc
