/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera, Sergei Bastrakov
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
#include "pmacc/mappings/kernel/AreaMappingMethods.hpp"
#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc
{
    /** Mapping between block indices and supercells in the given area for alpaka kernels
     *
     * The mapping covers the case of supercell-level parallelism between alpaka blocks.
     * The supercells can be processed concurrently in any order.
     *
     * This class makes a 1:1 mapping between supercells in the area and alpaka blocks.
     * A kernel must be launched with exactly getGridDim() blocks.
     * Each block must process a single supercell with index getSuperCellIndex(blockIndex).
     * Implementation is optimized for the standard areas in AreaType.
     *
     * In-block parallelism is independent of this mapping and is done by a kernel.
     * Naturally, this parallelism should also define block size, again independent of this mapping.
     *
     * This pattern is used in most kernels in particle-mesh codes.
     * Two most common parallel patterns are:
     *    - for particle or particle-grid operations:
     *      alpaka block per supercell with this mapping, thread-level parallelism between particles of a frame
     *    - for grid operations:
     *      alpaka block per supercell with this mapping, thread-level parallelism between cells of a supercell
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam T_MappingDescription mapping description type
     */
    template<uint32_t T_area, typename T_MappingDescription>
    class AreaMapping;

    template<uint32_t areaType, template<unsigned, class> class baseClass, unsigned DIM, class SuperCellSize_>
    class AreaMapping<areaType, baseClass<DIM, SuperCellSize_>> : public baseClass<DIM, SuperCellSize_>
    {
    public:
        typedef baseClass<DIM, SuperCellSize_> BaseClass;

        enum
        {
            AreaType = areaType,
            Dim = BaseClass::Dim
        };


        typedef typename BaseClass::SuperCellSize SuperCellSize;

        HINLINE AreaMapping(BaseClass base) : BaseClass(base)
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
            return AreaMappingMethods<areaType, DIM>::getGridDim(*this, this->getGridSuperCells());
        }

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index
         */
        HDINLINE DataSpace<DIM> getSuperCellIndex(const DataSpace<DIM>& blockIdx) const
        {
            return AreaMappingMethods<areaType, DIM>::getBlockIndex(*this, this->getGridSuperCells(), blockIdx);
        }
    };

    /** Construct an area mapping instance for the given area and description
     *
     * Currently always returns AreaMapping, but in principle could return a compatible type.
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam T_MappingDescription mapping description type
     */
    template<uint32_t T_area, typename T_MappingDescription>
    HINLINE auto makeAreaMapper(T_MappingDescription mappingDescription)
    {
        return AreaMapping<T_area, T_MappingDescription>{mappingDescription};
    }

} // namespace pmacc
