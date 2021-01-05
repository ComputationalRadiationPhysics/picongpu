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

#include "pmacc/types.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/mappings/kernel/StrideMappingMethods.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"

namespace pmacc
{
    template<uint32_t areaType, uint32_t stride, class baseClass>
    class StrideMapping;

    template<
        uint32_t areaType,
        uint32_t stride,
        template<unsigned, class>
        class baseClass,
        unsigned DIM,
        class SuperCellSize_>
    class StrideMapping<areaType, stride, baseClass<DIM, SuperCellSize_>> : public baseClass<DIM, SuperCellSize_>
    {
    public:
        typedef baseClass<DIM, SuperCellSize_> BaseClass;

        enum
        {
            AreaType = areaType,
            Dim = BaseClass::Dim,
            Stride = stride
        };


        typedef typename BaseClass::SuperCellSize SuperCellSize;

        HINLINE StrideMapping(BaseClass base) : BaseClass(base), offset()
        {
        }

        /**
         * Generate grid dimension information for kernel calls
         *
         * @return size of the grid
         */
        HINLINE DataSpace<DIM> getGridDim() const
        {
            return (StrideMappingMethods<areaType, DIM>::getGridDim(*this) - offset + (int) Stride - 1) / (int) Stride;
        }

        /**
         * Returns index of current logical block
         *
         * @param realSuperCellIdx current SuperCell index (block index)
         * @return mapped SuperCell index
         */
        HDINLINE DataSpace<DIM> getSuperCellIndex(const DataSpace<DIM>& realSuperCellIdx) const
        {
            const DataSpace<DIM> blockId((realSuperCellIdx * (int) Stride) + offset);
            return StrideMappingMethods<areaType, DIM>::shift(*this, blockId);
        }

        HDINLINE DataSpace<DIM> getOffset() const
        {
            return offset;
        }

        HDINLINE void setOffset(const DataSpace<DIM> offset)
        {
            this->offset = offset;
        }

        /** set mapper to next domain
         *
         * @return true if domain is valid, else false
         */
        HINLINE bool next()
        {
            int linearOffset = DataSpaceOperations<Dim>::map(DataSpace<DIM>::create(stride), offset);
            linearOffset++;
            offset = DataSpaceOperations<Dim>::map(DataSpace<DIM>::create(stride), linearOffset);

            return linearOffset < DataSpace<DIM>::create(stride).productOfComponents();
        }

    private:
        PMACC_ALIGN(offset, DataSpace<DIM>);
    };

} // namespace pmacc
