/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera
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

#include <stdexcept>
#include "pmacc/verify.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/math/Vector.hpp"

namespace pmacc
{
    /**
     * Abstracts logical block information from block variables.
     *
     * @tparam DIM dimension for grid/blocks
     * @tparam SuperCellSize mapper class for logical grid information
     */

    template<unsigned DIM, class SuperCellSize_>
    class MappingDescription
    {
    public:
        enum
        {
            Dim = DIM
        };

        typedef SuperCellSize_ SuperCellSize;

        /** constructor
         *
         * @param localGridCells number of cells in the local value (including guarding cells)
         * @param guardingSuperCells number of **supercells** within the guard
         */
        MappingDescription(
            DataSpace<DIM> localGridCells,
            DataSpace<DIM> guardingSuperCells = DataSpace<DIM>::create(0))
            : gridSuperCells(localGridCells / SuperCellSize::toRT())
            , /*block count per dimension*/
            guardingSuperCells(guardingSuperCells)
        {
            /* each dimension needs at least one supercell for the core and 2 * guardingSuperCells
             * (one supercell for the border and one for the guard) or it has no guarding and border
             * and contains only a core (this is allowed for local arrays which can not sync the
             * outer supercells with there neighbor MPI ranks.
             */
            for(uint32_t d = 0; d < DIM; ++d)
            {
                /*minimal 3 blocks are needed if we have guarding blocks*/
                int minBlock = std::min(gridSuperCells.x(), gridSuperCells.y());
                if(DIM == DIM3)
                {
                    minBlock = std::min(minBlock, gridSuperCells[2]);
                }
                PMACC_VERIFY(
                    (guardingSuperCells[d] == 0 && gridSuperCells[d] >= 1)
                    || gridSuperCells[d] >= 2 * guardingSuperCells[d] + 1);
            }
        }

        HDINLINE DataSpace<DIM> getGridSuperCells() const
        {
            return this->gridSuperCells;
        }

        HDINLINE DataSpace<DIM> getGuardingSuperCells() const
        {
            return guardingSuperCells;
        }

        HDINLINE void setGridSuperCells(DataSpace<DIM> superCellsCount)
        {
            gridSuperCells = superCellsCount;
        }

        /*! get the Coordinate of the root supercell in the hole simulation area
         * * root supercell in 2D LEFT+TOP | in 3D LEFT+TOP+FRONT
         * @param globaOffset cells
         * @return global index of the root supercell
         */
        HINLINE DataSpace<DIM> getRootSuperCellCoordinate(const DataSpace<DIM> globalOffset) const
        {
            return globalOffset / SuperCellSize::toRT();
        }

        HDINLINE DataSpace<DIM> getSuperCellSize() const
        {
            return SuperCellSize::toRT();
        }

        HDINLINE GridLayout<DIM> getGridLayout() const
        {
            return GridLayout<DIM>(
                SuperCellSize::toRT() * (gridSuperCells - 2 * guardingSuperCells),
                SuperCellSize::toRT() * guardingSuperCells);
        }

        HINLINE DataSpace<DIM> getGlobalSuperCells() const
        {
            return Environment<DIM>::get().GridController().getGpuNodes() * (gridSuperCells - 2 * guardingSuperCells);
        }


    protected:
        //\todo: keine Eigenschaft einer Zelle
        PMACC_ALIGN(gridSuperCells, DataSpace<DIM>);
        PMACC_ALIGN(guardingSuperCells, DataSpace<DIM>);
    };

} // namespace pmacc
