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
#include "pmacc/mappings/kernel/AreaMappingMethods.hpp"
#include "pmacc/mappings/kernel/MapperConcept.hpp"
#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc
{
    /** Mapping from block indices to supercells in the given area for alpaka kernels
     *
     * Adheres to the MapperConcept.
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
        using BaseClass = baseClass<DIM, SuperCellSize_>;

        enum
        {
            AreaType = areaType,
            Dim = BaseClass::Dim
        };


        using SuperCellSize = typename BaseClass::SuperCellSize;

        HINLINE AreaMapping(BaseClass base) : BaseClass(base)
        {
        }

        AreaMapping(const AreaMapping&) = default;

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
         * @tparam T_origin Which origin (CORE/BORDER/GUARD) to return supercell index relative to (default: GUARD)
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        template<uint32_t T_origin = GUARD>
        HDINLINE DataSpace<DIM> getSuperCellIndex(const DataSpace<DIM>& blockIdx) const
        {
            auto result
                = AreaMappingMethods<areaType, DIM>::getSuperCellIndex(*this, this->getGridSuperCells(), blockIdx);
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
    };

    /** Construct an area mapper instance for the given standard area and description
     *
     * Adheres to the MapperFactoryConcept.
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     */
    template<uint32_t T_area>
    struct AreaMapperFactory
    {
        /** Construct an area mapper object
         *
         * @tparam T_MappingDescription mapping description type
         *
         * @param mappingDescription mapping description
         *
         * @return an object adhering to the AreaMapping concept
         */
        template<typename T_MappingDescription>
        HINLINE auto operator()(T_MappingDescription mappingDescription) const
        {
            return AreaMapping<T_area, T_MappingDescription>{mappingDescription};
        }
    };

    /** Construct an area mapper instance for the given area and description
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam T_MappingDescription mapping description type
     *
     * @param mappingDescription mapping description
     */
    template<uint32_t T_area, typename T_MappingDescription>
    HINLINE auto makeAreaMapper(T_MappingDescription mappingDescription)
    {
        return AreaMapperFactory<T_area>{}(mappingDescription);
    }

} // namespace pmacc
