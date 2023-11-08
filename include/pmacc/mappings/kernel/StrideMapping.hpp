/* Copyright 2013-2023 Felix Schmitt, Heiko Burau, Rene Widera, Sergei Bastrakov
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
    /** Strided mapping from block indices to supercells in the given area for alpaka kernels
     *
     * Adheres to the MapperConcept.
     *
     * The mapped area is subdivided into stride^dim non-intersecting subareas (some may be empty).
     * A subarea is an intersection of the area and an integer lattice with given stride in all directions.
     * Each subarea has a unique offset relative to area start.
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam T_stride stride value, same for all directions
     * @tparam baseClass mapping description type
     */
    template<uint32_t areaType, uint32_t T_stride, class baseClass>
    class StrideMapping;

    template<
        uint32_t areaType,
        uint32_t T_stride,
        template<unsigned, class>
        class baseClass,
        unsigned DIM,
        class SuperCellSize_>
    class StrideMapping<areaType, T_stride, baseClass<DIM, SuperCellSize_>> : public baseClass<DIM, SuperCellSize_>
    {
    public:
        using BaseClass = baseClass<DIM, SuperCellSize_>;

        //! Stride value
        static constexpr uint32_t stride = T_stride;

        enum
        {
            AreaType = areaType,
            Dim = BaseClass::Dim,
        };


        using SuperCellSize = typename BaseClass::SuperCellSize;

        HINLINE StrideMapping(BaseClass base) : BaseClass(base), offset()
        {
        }

        StrideMapping(const StrideMapping&) = default;

        /** Generate grid dimension information for alpaka kernel calls
         *
         * A kernel using this mapping must use exacly the returned number of blocks
         *
         * @return number of blocks in a grid
         */
        HINLINE DataSpace<DIM> getGridDim() const
        {
            return (StrideMappingMethods<areaType, DIM>::getGridDim(*this) - offset + stride - 1) / stride;
        }

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @tparam T_origin Which origin (CORE/BORDER/GUARD) to return supercell index relative to (default: GUARD)
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        template<uint32_t T_origin = GUARD>
        HDINLINE DataSpace<DIM> getSuperCellIndex(const DataSpace<DIM>& blockIdx) const
        {
            const DataSpace<DIM> blockId((blockIdx * stride) + offset);
            auto result = StrideMappingMethods<areaType, DIM>::shift(*this, blockId);
            if constexpr(T_origin == CORE)
            {
                result = result - 2 * this->getGuardingSuperCells();
            }
            if constexpr(T_origin == BORDER)
            {
                result = result - this->getGuardingSuperCells();
            }
            return result;
        }

        HDINLINE DataSpace<DIM> getOffset() const
        {
            return offset;
        }

        HDINLINE void setOffset(const DataSpace<DIM> offset)
        {
            this->offset = offset;
        }

        /** Set mapper to next non-empty subarea
         *
         * Note: this function has no HINLINE as it is recursive and so cannot be force-inlined.
         *
         * @return whether the whole area was processed
         */
        bool next()
        {
            int linearOffset = DataSpaceOperations<Dim>::map(DataSpace<DIM>::create(stride), offset);
            linearOffset++;
            offset = DataSpaceOperations<Dim>::map(DataSpace<DIM>::create(stride), linearOffset);
            /* First check if everything is processed to have a recursion stop condition.
             * Then if the new grid dim has 0 size, immediately go to the next state.
             * This way to guarantee that when next() returned true, a grid dim is valid.
             */
            if(linearOffset >= DataSpace<DIM>::create(stride).productOfComponents())
                return false;
            if(getGridDim().productOfComponents() == 0)
                return next();
            return true;
        }

    private:
        PMACC_ALIGN(offset, DataSpace<DIM>);
    };

    /** Construct a strided area mapper instance for the given standard area, stride, and description
     *
     * Adheres to the MapperFactoryConcept.
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam stride stride value, same for all directions
     */
    template<uint32_t T_area, uint32_t T_stride>
    struct StrideAreaMapperFactory
    {
        /** Construct a strided area mapper object
         *
         * @tparam T_MappingDescription mapping description type
         *
         * @param mappingDescription mapping description
         *
         * @return an object adhering to the StridedMapping concept
         */
        template<typename T_MappingDescription>
        HINLINE auto operator()(T_MappingDescription mappingDescription) const
        {
            return StrideMapping<T_area, T_stride, T_MappingDescription>{mappingDescription};
        }
    };

    /** Construct a strided area mapper instance for the given area, stride, and description
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam stride stride value, same for all directions
     * @tparam T_MappingDescription mapping description type
     *
     * @param mappingDescription mapping description
     */
    template<uint32_t T_area, uint32_t T_stride, typename T_MappingDescription>
    HINLINE auto makeStrideAreaMapper(T_MappingDescription mappingDescription)
    {
        return StrideAreaMapperFactory<T_area, T_stride>{}(mappingDescription);
    }

} // namespace pmacc
