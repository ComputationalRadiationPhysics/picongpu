/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
