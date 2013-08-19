/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera
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
 
#ifndef AREAMAPPING_H
#define	AREAMAPPING_H

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "mappings/kernel/AreaMappingMethods.hpp"

namespace PMacc
{

    template<uint32_t areaType, class baseClass>
    class AreaMapping;

    template<
    uint32_t areaType,
    template<unsigned, class> class baseClass,
    unsigned DIM,
    class SuperCellSize_
    >
    class AreaMapping<areaType, baseClass<DIM, SuperCellSize_> > : public baseClass<DIM, SuperCellSize_>
    {
    public:
        typedef baseClass<DIM, SuperCellSize_> BaseClass;

        enum
        {
            AreaType = areaType, Dim = BaseClass::Dim
        };


        typedef typename BaseClass::SuperCellSize SuperCellSize;

        HINLINE AreaMapping(BaseClass base) : BaseClass(base)
        {
        }

        /**
         * Generates cuda gridDim information for kernel call.
         *
         * @return dim3 with gridDim information
         */
        HINLINE DataSpace<DIM> getGridDim()
        {
            return this->reduce(AreaMappingMethods<areaType, DIM>::getGridDim(*this,
                                                                        this->getGridSuperCells()));
        }

        /**
         * Returns index of current logical block, depending on current cuda block id.
         *
         * @param _blockIdx current cuda block id (blockIdx)
         * @return current logical block index
         */
        DINLINE DataSpace<DIM> getSuperCellIndex(const DataSpace<DIM>& realSuperCellIdx)
        {
            return AreaMappingMethods<areaType, DIM>::getBlockIndex(*this,
                                                                    this->getGridSuperCells(),
                                                                    extend(realSuperCellIdx));
        }

    };


} // namespace PMacc



#endif	/* AREAMAPPING_H */

