/* Copyright 2021-2023 Sergei Bastrakov
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
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/mappings/kernel/IntervalMapping.hpp"
#include "pmacc/mappings/kernel/StrideIntervalMapping.hpp"
#include "pmacc/mappings/kernel/StrideMapping.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** Wrapper of a mapper factory to construct a strided mapper for the area defined by the given factory
     *
     * Adheres to the MapperFactoryConcept.
     *
     * @tparam T_BaseMapperFactory factory type that constructs a mapper defining the area
     * @tparam stride stride value, same for all directions
     */
    template<typename T_BaseMapperFactory, uint32_t T_stride>
    struct StrideMapperFactory
    {
        /** Construct a stride factory for the are defined by the given base factory
         *
         * @param baseFactory factory instance that constructs a mapper defining the area
         */
        StrideMapperFactory(T_BaseMapperFactory const& baseFactory);

        /** Construct a strided area mapper object
         *
         * @tparam T_MappingDescription mapping description type
         *
         * @param mappingDescription mapping description
         *
         * @return an object adhering to the StridedMapping concept
         */
        template<typename T_MappingDescription>
        HINLINE auto operator()(T_MappingDescription mappingDescription) const;
    };

    /** Wrapper of a area mapper factory to construct a strided mapper for the area
     *
     * Adheres to the MapperFactoryConcept.
     *
     * @tparam T_area area, a value from type::AreaType or a sum of such values
     * @tparam stride stride value, same for all directions
     */
    template<uint32_t T_area, uint32_t T_stride>
    struct StrideMapperFactory<AreaMapperFactory<T_area>, T_stride>
    {
        /** Construct a stride factory for the are defined by the given base factory
         *
         * @param baseFactory factory instance that constructs a mapper defining the area
         */
        StrideMapperFactory(AreaMapperFactory<T_area> const&)
        {
        }

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
            return makeStrideAreaMapper<T_area, T_stride>(mappingDescription);
        }
    };

    /** Wrapper of a area mapper factory to construct a strided mapper for the area
     *
     * Adheres to the MapperFactoryConcept.
     *
     * @tparam T_dim dimensionality of mappers to be constructed
     * @tparam T_stride stride value, same for all directions
     */
    template<uint32_t T_dim, uint32_t T_stride>
    struct StrideMapperFactory<IntervalMapperFactory<T_dim>, T_stride>
    {
        /** Construct a stride factory for the are defined by the given base factory
         *
         * @param baseFactory factory instance that constructs a mapper defining the area
         */
        StrideMapperFactory(IntervalMapperFactory<T_dim> const& factory)
            : beginSupercell(factory.beginSupercell)
            , numSupercells(factory.numSupercells)
        {
        }

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
            return makeStrideIntervalMapper<T_stride>(mappingDescription, beginSupercell, numSupercells);
        }

    private:
        //! The first supercell index of the interval, including guards, in the constructed objects
        DataSpace<T_dim> const beginSupercell;

        //! Number of supercells in the interval along each direction in the constructed objects
        DataSpace<T_dim> const numSupercells;
    };

} // namespace pmacc
