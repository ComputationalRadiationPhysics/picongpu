/**
 * Copyright 2013-2015 Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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


#pragma once

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "mappings/kernel/BorderMappingMethods.hpp"
#include <stdexcept>

namespace PMacc
{

    /**
     * This maps onto the border to 1 direction (e.g. TOP, BOTTOM, ...)
     * The area is basically the same as the surrounding guard region
     * but on the border. Therefore only the "main" directions are allowed
     * (no diagonal ones)
     *
     * @tparam T_BaseClass base class for mapping, should be MappingDescription
     */
    template<class T_BaseClass>
    class BorderMapping;

    template<
        template<unsigned, class> class T_BaseClass,
        unsigned T_dim,
        class T_SuperCellSize
    >
    class BorderMapping<T_BaseClass<T_dim, T_SuperCellSize> >
    {
    public:
        typedef T_BaseClass<T_dim, T_SuperCellSize> BaseClass;

        enum
        {
            Dim = BaseClass::Dim,
            AreaType = BORDER
        };
        typedef DataSpace<Dim> DimDataSpace;

        typedef typename BaseClass::SuperCellSize SuperCellSize;

        /**
         * Constructor.
         *
         * @param base object of base class baseClass (see template parameters)
         * @param direction direction to map to
         */
        HINLINE BorderMapping(const BaseClass& base, PMacc::ExchangeType direction)
        {
            DimDataSpace relDir = Mask::getRelativeDirections<Dim>(direction);
            int dirCt = 0;
            for(int i = 0; i < Dim; ++i)
                if(relDir[i] != 0)
                    ++dirCt;
            /* Only exactly 1 direction is allowed */
            if(dirCt != 1)
                throw std::logic_error("Invalid direction");
            gridDim = BorderMappingMethods<Dim>::getGridDim(base, direction);
            blockOffset = BorderMappingMethods<Dim>::getBlockOffset(base, direction);
        }

        /**
         * Generate grid dimension information for kernel calls
         *
         * @return size of the grid
         */
        HDINLINE DimDataSpace getGridDim() const
        {
            return gridDim;
        }

        /**
         * Returns index of current logical block
         *
         * @param realSuperCellIdx current SuperCell index (block index)
         * @return mapped SuperCell index
         */
        HDINLINE DimDataSpace getSuperCellIndex(const DimDataSpace& realSuperCellIdx) const
        {
            return realSuperCellIdx + blockOffset;
        }
    private:
        PMACC_ALIGN(gridDim, DimDataSpace);
        PMACC_ALIGN(blockOffset, DimDataSpace);
    };
} // namespace PMacc
