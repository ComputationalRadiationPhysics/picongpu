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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"
#include "pmacc/mappings/kernel/MapperConcept.hpp"
#include "pmacc/mappings/kernel/StrideMappingMethods.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** Mapping from block indices to supercells in the given strided area for alpaka kernels
     *
     * Adheres to the MapperConcept.
     *
     * The mapped area is an intersection of T_area and an integer lattice with given stride in all directions
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam stride stride value
     * @tparam baseClass mapping description type
     */
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

        /** Generate grid dimension information for alpaka kernel calls
         *
         * A kernel using this mapping must use exacly the returned number of blocks
         *
         * @return number of blocks in a grid
         */
        HINLINE DataSpace<DIM> getGridDim() const
        {
            return (StrideMappingMethods<areaType, DIM>::getGridDim(*this) - offset + (int) Stride - 1) / (int) Stride;
        }

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        HDINLINE DataSpace<DIM> getSuperCellIndex(const DataSpace<DIM>& blockIdx) const
        {
            const DataSpace<DIM> blockId((blockIdx * (int) Stride) + offset);
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
