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

#include <stdint.h>
#include "pmacc/math/vector/Int.hpp"
#include "pmacc/math/vector/Size_t.hpp"

namespace pmacc
{
    namespace zone
    {
        namespace tag
        {
            struct SphericZone
            {
            };
        } // namespace tag

        /* spheric (no holes), cartesian zone
         *
         * \tparam T_dim dimension of the zone
         *
         * This is a zone which is simply described by a size and a offset.
         *
         */
        template<int T_dim>
        struct SphericZone
        {
            typedef tag::SphericZone tag;
            static constexpr int dim = T_dim;
            math::Size_t<dim> size;
            math::Int<dim> offset;

            HDINLINE SphericZone()
            {
            }
            HDINLINE SphericZone(const math::Size_t<dim>& size) : size(size), offset(math::Int<dim>::create(0))
            {
            }
            HDINLINE SphericZone(const math::Size_t<dim>& size, const math::Int<dim>& offset)
                : size(size)
                , offset(offset)
            {
            }

            /* Returns whether pos is within the zone */
            HDINLINE bool within(const pmacc::math::Int<T_dim>& pos) const
            {
                bool result = true;
                for(int i = 0; i < T_dim; i++)
                    if((pos[i] < offset[i]) || (pos[i] >= offset[i] + (int) size[i]))
                        result = false;
                return result;
            }
        };

    } // namespace zone
} // namespace pmacc
