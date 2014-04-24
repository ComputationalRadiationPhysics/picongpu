/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MAPPINGDESCRIPTION_H
#define	MAPPINGDESCRIPTION_H

#include <builtin_types.h>
#include <stdexcept>
#include "dimensions/DataSpace.hpp"
#include "dimensions/DataSpaceOperations.hpp"
#include "mappings/simulation/GridController.hpp"
#include "dimensions/GridLayout.hpp"
#include "mappings/kernel/CudaGridDimRestrictions.hpp"

namespace PMacc
{

/**
 * Abstracts logical block information from cuda block variables.
 *
 * @tparam DIM dimension for grid/blocks
 * @tparam SuperCellSize mapper class for logical grid information
 */

template<unsigned DIM, class SuperCellSize_>
class MappingDescription :
public CudaGridDimRestrictions<DIM>
{
public:

    enum
    {
        Dim = DIM
    };

    typedef SuperCellSize_ SuperCellSize;

    MappingDescription(DataSpace<DIM> localGridCells = DataSpace<DIM> (),
                       uint32_t borderSuperCells = 0,
                       uint32_t guardingSuperCells = 0) :
    gridSuperCells(localGridCells / SuperCellSize::getDataSpace()), /*block count per dimension*/
    guardingSuperCells(guardingSuperCells),
    borderSuperCells(borderSuperCells)
    {
        /*minimal 3 blocks are needed if we have guarding blocks*/
        int minBlock = std::min(gridSuperCells.x(), gridSuperCells.y());
        if (DIM == DIM3)
        {
            minBlock = std::min(minBlock, gridSuperCells[2]);
        }
        assert((guardingSuperCells == 0) || (minBlock >= 3));
        /*border block count must smaller or equal to core blocks count*/
        assert(borderSuperCells <= (minBlock - (2 * guardingSuperCells)));
    }

    HDINLINE DataSpace<DIM> getGridSuperCells() const
    {
        return this->gridSuperCells;
    }

    HDINLINE uint32_t getBorderSuperCells() const
    {
        return borderSuperCells;
    }

    HDINLINE uint32_t getGuardingSuperCells() const
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
    HINLINE DataSpace<DIM> getRootSuperCellCoordinate(const DataSpace<DIM> globalOffset)
    {
        return globalOffset/SuperCellSize::getDataSpace();
    }

    HDINLINE DataSpace<DIM> getSuperCellSize() const
    {
        return SuperCellSize::getDataSpace();
    }

    HDINLINE GridLayout<DIM> getGridLayout() const
    {
        return GridLayout<DIM > (SuperCellSize::getDataSpace()*(gridSuperCells - 2 * guardingSuperCells), SuperCellSize::getDataSpace() * guardingSuperCells);
    }

    HINLINE DataSpace<DIM> getGlobalSuperCells() const
    {
        return Environment<DIM>::get().GridController().getGpuNodes() * (gridSuperCells - 2 * guardingSuperCells);
    }


protected:

    //\todo: keine Eigenschaft einer Zelle
    PMACC_ALIGN(gridSuperCells, DataSpace<DIM>);
    PMACC_ALIGN(guardingSuperCells, uint32_t);
    PMACC_ALIGN(borderSuperCells, uint32_t);

};


} // namespace PMacc



#endif	/* MappingDescription_H */

