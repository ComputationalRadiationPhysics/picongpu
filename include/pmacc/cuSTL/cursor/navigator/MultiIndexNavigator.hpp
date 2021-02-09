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

#include "tag.hpp"
#include "pmacc/math/vector/Int.hpp"
#include "pmacc/cuSTL/cursor/traits.hpp"


namespace pmacc
{
    namespace cursor
    {
        template<int T_dim>
        struct MultiIndexNavigator
        {
            typedef tag::MultiIndexNavigator tag;
            static constexpr int dim = T_dim;

            template<typename MultiIndex>
            HDINLINE MultiIndex operator()(const MultiIndex& index, const math::Int<dim>& jump) const
            {
                return index + jump;
            }
        };

        namespace traits
        {
            template<int T_dim>
            struct dim<MultiIndexNavigator<T_dim>>
            {
                static constexpr int value = T_dim;
            };

        } // namespace traits

    } // namespace cursor
} // namespace pmacc
