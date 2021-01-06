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
    namespace cursor
    {
        template<typename Marker>
        struct MarkerAccessor
        {
            typedef const Marker type;
            /** returns the cursor's marker.
             *
             * Here a copy of marker is returned because the cursor object
             * could be a temporary object. Therefore any reference or const-reference
             * of marker is dangerous. If you want to have a reference to marker use e.g.
             * FunctorAccessor or Cursor::getMarker().
             */
            HDINLINE
            type operator()(const Marker& marker) const
            {
                return marker;
            }
        };

    } // namespace cursor
} // namespace pmacc
