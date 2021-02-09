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

#include "Cursor.hpp"
#include "accessor/MarkerAccessor.hpp"
#include "navigator/MultiIndexNavigator.hpp"
#include "pmacc/math/vector/Int.hpp"

namespace pmacc
{
    namespace cursor
    {
        /** construct a cursor where accessing means getting the current position
         * in terms of an 2D, 3D, ... index. Usefull to obtain for example the current cell index.
         * \tparam dim Dimension of the index (say: int-vector)
         * \param idx Initial index value
         * \return cursor with the behavior mentioned above
         */
        template<int dim>
        HDINLINE cursor::Cursor<cursor::MarkerAccessor<math::Int<dim>>, MultiIndexNavigator<dim>, math::Int<dim>>
        make_MultiIndexCursor(const math::Int<dim>& idx = math::Int<dim>::create(0))
        {
            return make_Cursor(cursor::MarkerAccessor<math::Int<dim>>(), MultiIndexNavigator<dim>(), idx);
        }

    } // namespace cursor
} // namespace pmacc
