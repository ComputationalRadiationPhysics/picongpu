/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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

namespace pmacc
{
    /**
     * Describes layout of a DIM-dimensional data grid including the actual grid and optional guards.
     *
     * @tparam DIM dimension of the grid
     */
    template<unsigned DIM>
    class GridLayout
    {
    public:
        HDINLINE GridLayout() : dataSpace(DataSpace<DIM>::create(1)), guard(DataSpace<DIM>::create(0))
        {
        }

        /**
         * constructor
         * @param dataSpace DataSpace defining size of the layout (native loacal simulation area whithout any guarding)
         * @param guard DataSpace defining size of the guard cells. Guard is added to actual grid (dataSpace). Will be
         * initialized to 0.
         */
        HDINLINE GridLayout(const DataSpace<DIM>& dataSpace, DataSpace<DIM> guard = DataSpace<DIM>())
            : dataSpace(dataSpace)
            , guard(guard)
        {
        }

        /**
         * returns the DataSpace for the data
         * (include guarding for overlap neighbor areas)
         * @return the data DataSpace
         */
        HDINLINE DataSpace<DIM> getDataSpace() const
        {
            return dataSpace + guard + guard;
        }

        HDINLINE DataSpace<DIM> getDataSpaceWithoutGuarding() const
        {
            return dataSpace;
        }

        /**
         * returns the DataSpace for the guard
         * @return the guard DataSpace
         */
        HDINLINE DataSpace<DIM> getGuard() const
        {
            return guard;
        }

    private:
        DataSpace<DIM> dataSpace;
        DataSpace<DIM> guard;
    };

} // namespace pmacc
