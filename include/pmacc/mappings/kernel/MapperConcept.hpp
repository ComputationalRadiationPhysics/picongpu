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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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

namespace pmacc
{
    /** Concept for mapper from block indices to supercells in the given area for alpaka kernels
     *
     * The mapping covers the case of supercell-level parallelism between alpaka blocks.
     * The supercells can be processed concurrently in any order.
     * This class is not used directly, but defines a concept for such mappers.
     *
     * Mapper provides a 1:1 mapping from supercells in the area to alpaka blocks.
     * (Since it is 1:1, the mapping is invertible, but only this direction is provided.)
     * Dimensionality of the area indices and block indices is the same.
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
     * @tparam T_areaType parameter describing area to be mapped (depends on mapper type)
     * @tparam T_MappingDescription mapping description type, base class for MapperConcept
     * @tparam T_dim dimensionality of area and block indices
     * @tparam T_SupercellSize compile-time supercell size
     */
    template<
        uint32_t T_areaType,
        template<unsigned, typename>
        class T_MappingDescription,
        unsigned T_dim,
        typename T_SupercellSize>
    class MapperConcept : public T_MappingDescription<T_dim, T_SupercellSize>
    {
    public:
        //! Base class
        using BaseClass = T_MappingDescription<T_dim, T_SupercellSize>;

        //! Compile-time super cell size
        using SuperCellSize = typename BaseClass::SuperCellSize;

        /** Create a mapper instance
         *
         * @param base base class instance
         */
        HINLINE MapperConcept(T_MappingDescription<T_dim, T_SupercellSize> base);

        /** Generate grid dimension information for alpaka kernel calls
         *
         * A kernel using this mapping must use exacly the returned number of blocks
         *
         * @return number of blocks in a grid
         */
        HINLINE DataSpace<T_dim> getGridDim() const;

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        HDINLINE DataSpace<T_dim> getSuperCellIndex(const DataSpace<T_dim>& blockIdx) const;
    };

    /** Concept for mapper factory
     *
     * Defines interface for implementations of such factories.
     * (A user-provided implementation is needed for user-defined areas.)
     */
    class MapperFactoryConcept
    {
    public:
        /** Construct an mapper object using the given base class instance
         *
         * @tparam T_MappingDescription mapping description type
         *
         * @param mappingDescription mapping description
         *
         * @return an object adhering to the MapperConcept, inheriting T_MappingDescription
         */
        template<typename T_MappingDescription>
        HINLINE auto operator()(T_MappingDescription mappingDescription) const;
    };

} // namespace pmacc
