/* Copyright 2024 Rene Widera
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
#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc
{
    /** Mapping from block indices to supercells in the given area for alpaka kernels
     *
     * @tparam T_BaseMapper Mapper which should be mapped to a one dimension domain.
     *                      Supports non contiguous mapper e.g. StrideMapper too.
     */
    template<typename T_BaseMapper>
    class RangeMapping : T_BaseMapper
    {
    public:
        using BaseClass = T_BaseMapper;
        using BaseClass::getGuardingSuperCells;
        using BaseClass::getSuperCellSize;

        static constexpr uint32_t Dim = BaseClass::Dim;

        using SuperCellSize = typename BaseClass::SuperCellSize;

        /** constructor
         *
         * @param base mapper to be wrapped and linearized
         * @param begin index (included) of the first elements
         * @param end index one behind the last valid index, -1 will automaticlally derive the end from the wrapped
         * mapper
         */
        HINLINE RangeMapping(BaseClass const& base, uint32_t const begin, uint32_t const end)
            : BaseClass(base)
            , baseGridDim(base.getGridDim())
            , userDefinedBeginIdx(begin)
            , userDefinedEndIdx(end)
        {
            clampRange();
        }

        RangeMapping(RangeMapping const&) = default;

        /** Generate grid dimension information for alpaka kernel calls
         *
         * A kernel using this mapping must use exactly the returned number of blocks.
         * The range will automatically clamped to fit into the N-dimensional block range of the wrapped mapper.
         * To support mapper with dynamic grid dimensions the range e.g. StrideMapper each call to this method is
         * updating the range.
         *
         * @return number of blocks in a grid
         */
        HINLINE DataSpace<DIM1> getGridDim()
        {
            clampRange();
            return {size()};
        }

        /** Number of elements described by the range
         *
         * @return  size of the range
         */
        HDINLINE uint32_t size() const
        {
            return endIdx - beginIdx;
        }

        /** Supercell index of the first element */
        HDINLINE uint32_t begin() const
        {
            return beginIdx;
        }

        /** Supercell index of the element behind the last valid element */
        HDINLINE uint32_t end() const
        {
            return endIdx;
        }

        /** Supercell index of tha last valid element */
        HDINLINE uint32_t last() const
        {
            return endIdx == 0u ? 0u : endIdx - 1u;
        }

        /** N-dimensional supercell index of the first element */
        HDINLINE DataSpace<Dim> beginND() const
        {
            return pmacc::math::mapToND(baseGridDim, beginIdx);
        }

        /** N-dimensional supercell index of the element behind the last valid element */
        HDINLINE DataSpace<Dim> endND() const
        {
            return pmacc::math::mapToND(baseGridDim, endIdx);
        }

        /** N-dimensional supercell index of tha last valid element */
        HDINLINE DataSpace<Dim> lastND() const
        {
            return pmacc::math::mapToND(baseGridDim, last());
        }

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        HDINLINE DataSpace<Dim> getSuperCellIndex(const DataSpace<DIM1>& blockIdx) const
        {
            auto const blockIdxNDim = pmacc::math::mapToND(baseGridDim, static_cast<int>(beginIdx + blockIdx.x()));
            return BaseClass::getSuperCellIndex(blockIdxNDim);
        }

        /** 1-dimensional supercell index */
        HDINLINE int superCellIdx(DataSpace<DIM1> const& blockIdx) const
        {
            return blockIdx.x();
        }

    private:
        /** clamp range to valid values based on the wrapped mapper
         *
         * Calling this method is updating baseGridDim to the grid dimensions of the wrapped mapper.
         */
        HINLINE void clampRange()
        {
            /* Update the base grid size in case it is change over time if the mapper instance is used
             * multiple times e.g. StridingMapping
             */
            baseGridDim = BaseClass::getGridDim();
            uint32_t const baseMapperLast = static_cast<uint32_t>(baseGridDim.productOfComponents());
            endIdx = userDefinedEndIdx == uint32_t(-1) ? baseMapperLast : userDefinedEndIdx;
            endIdx = std::min(endIdx, baseMapperLast);
            beginIdx = std::min(userDefinedBeginIdx, endIdx);
        }

        DataSpace<Dim> baseGridDim;
        uint32_t beginIdx = 0u;
        uint32_t endIdx = 0u;
        uint32_t const userDefinedBeginIdx = 0u;
        uint32_t const userDefinedEndIdx = uint32_t(-1);
    };

    /** Construct an area mapper instance for the given area and description
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam T_MappingDescription mapping description type
     *
     * @param baseMapper mapper to be wrapped and linearized
     * @param begin index (included) of the first elements
     * @param end index one behind the last valid index, -1 will automaticlally derive the end from the wrapped mapper
     */
    template<typename T_BaseMapper>
    HINLINE auto makeRangeMapper(T_BaseMapper baseMapper, uint32_t begin = 0u, uint32_t end = uint32_t(-1))
    {
        return RangeMapping(baseMapper, begin, end);
    }

} // namespace pmacc
