/* Copyright 2013-2023 Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig,
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
     * Describes layout of a T_dim-dimensional data grid including the actual grid and optional guards.
     *
     * @tparam T_dim dimension of the grid
     */
    template<unsigned T_dim>
    class GridLayout
    {
    public:
        HDINLINE GridLayout() : m_sizeND(DataSpace<T_dim>::create(1)), m_guardSizeND(DataSpace<T_dim>::create(0))
        {
        }

        /**
         * constructor
         * @param dataSpace DataSpace defining size of the layout (native loacal simulation area whithout any guarding)
         * @param guard DataSpace defining size of the guard cells. Guard is added to actual grid (dataSpace). Will be
         * initialized to 0.
         */
        HDINLINE GridLayout(DataSpace<T_dim> const& sizeND, DataSpace<T_dim> const& guardSizeND = DataSpace<T_dim>())
            : m_sizeND(sizeND)
            , m_guardSizeND(guardSizeND)
        {
        }

        /** N-dimensional size of the domain
         *
         * @return number of cells per dimension including guard cells
         */
        HDINLINE DataSpace<T_dim> sizeND() const
        {
            return m_sizeND + m_guardSizeND + m_guardSizeND;
        }

        /** N-dimensional size of the domain
         *
         * @return number of cells per dimension without guard cells
         */
        HDINLINE DataSpace<T_dim> sizeWithoutGuardND() const
        {
            return m_sizeND;
        }

        /** N-dimensional size of the guard
         *
         * @return number of cells in the guard area
         */
        HDINLINE DataSpace<T_dim> guardSizeND() const
        {
            return m_guardSizeND;
        }

    private:
        DataSpace<T_dim> m_sizeND;
        DataSpace<T_dim> m_guardSizeND;
    };

} // namespace pmacc
