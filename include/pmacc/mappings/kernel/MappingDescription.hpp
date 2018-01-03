/* Copyright 2013-2018 Felix Schmitt, Heiko Burau, Rene Widera
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

    MappingDescription(DataSpace<DIM> localGridCells = DataSpace<DIM> (),
                       int borderSuperCells = 0,
                       int guardingSuperCells = 0) :
    gridSuperCells(localGridCells / SuperCellSize::toRT()), /*block count per dimension*/
    guardingSuperCells(guardingSuperCells),
    borderSuperCells(borderSuperCells)
    {
        /*minimal 3 blocks are needed if we have guarding blocks*/
        int minBlock = std::min(gridSuperCells.x(), gridSuperCells.y());
        if (DIM == DIM3)
        {
            minBlock = std::min(minBlock, gridSuperCells[2]);
        }
        PMACC_VERIFY((guardingSuperCells == 0) || (minBlock >= 3));
        /*border block count must smaller or equal to core blocks count*/
        PMACC_VERIFY(borderSuperCells <= (minBlock - (2 * guardingSuperCells)));
    }

    HDINLINE DataSpace<DIM> getGridSuperCells() const
    {
        return this->gridSuperCells;
    }

    HDINLINE int getBorderSuperCells() const
    {
        return borderSuperCells;
    }

    HDINLINE int getGuardingSuperCells() const
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
        return globalOffset/SuperCellSize::toRT();
    }

    HDINLINE DataSpace<DIM> getSuperCellSize() const
    {
        return SuperCellSize::toRT();
    }

    HDINLINE GridLayout<DIM> getGridLayout() const
    {
        return GridLayout<DIM > (SuperCellSize::toRT()*(gridSuperCells - 2 * guardingSuperCells), SuperCellSize::toRT() * guardingSuperCells);
    }

    HINLINE DataSpace<DIM> getGlobalSuperCells() const
    {
        return Environment<DIM>::get().GridController().getGpuNodes() * (gridSuperCells - 2 * guardingSuperCells);
    }


protected:

    //\todo: keine Eigenschaft einer Zelle
    PMACC_ALIGN(gridSuperCells, DataSpace<DIM>);
    PMACC_ALIGN(guardingSuperCells, int);
    PMACC_ALIGN(borderSuperCells, int);

};

} // namespace pmacc
