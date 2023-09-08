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
#include "pmacc/mappings/kernel/MapperConcept.hpp"
#include "pmacc/types.hpp"

#include <cstdint>

namespace pmacc
{
    /** Mapping from block indices to supercells in the given interval for alpaka kernels
     *
     * An interval is a T_dim-dimensional contiguous Cartesian range.
     * Parameters of the inverval are defined at runtime, unlike most other mappers.
     *
     * Adheres to the MapperConcept.
     *
     * @tparam T_MappingDescription mapping description type
     */
    template<typename T_MappingDescription>
    class IntervalMapping;

    template<template<unsigned, class> class T_MappingDescription, unsigned T_dim, typename T_SuperCellSize>
    class IntervalMapping<T_MappingDescription<T_dim, T_SuperCellSize>>
        : public T_MappingDescription<T_dim, T_SuperCellSize>
    {
    public:
        //! Base class
        using BaseClass = T_MappingDescription<T_dim, T_SuperCellSize>;

        //! Compile-time super cell size
        using SuperCellSize = typename BaseClass::SuperCellSize;

        /** Create a mapper instance
         *
         * @param base instance of the base class to be propagated
         * @param beginSupercell the first supercell index of the interval, including guards
         * @param numSupercells number of supercells in the interval along each direction
         */
        HINLINE IntervalMapping(
            BaseClass base,
            DataSpace<T_dim> const& beginSupercell,
            DataSpace<T_dim> const& numSupercells)
            : BaseClass(base)
            , beginSupercell(beginSupercell)
            , numSupercells(numSupercells)
        {
        }

        IntervalMapping(const IntervalMapping&) = default;

        /** Generate grid dimension information for alpaka kernel calls
         *
         * A kernel using this mapping must use exacly the returned number of blocks
         *
         * @return number of blocks in a grid
         */
        HINLINE DataSpace<T_dim> getGridDim() const
        {
            return numSupercells;
        }

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        HDINLINE DataSpace<T_dim> getSuperCellIndex(DataSpace<T_dim> const& blockIdx) const
        {
            return beginSupercell + blockIdx;
        }

    private:
        //! The first supercell index of the interval, including guards
        DataSpace<T_dim> const beginSupercell;

        //! Number of supercells in the interval along each direction
        DataSpace<T_dim> const numSupercells;
    };

    /** Construct an interval mapper instance for the given description
     *
     * Adheres to the MapperFactoryConcept.
     *
     * @tparam T_dim dimensionality of mappers to be constructed
     */
    template<unsigned T_dim>
    struct IntervalMapperFactory
    {
        /** Create a factory instance
         *
         * @param beginSupercell the first supercell index of the interval, including guards, in the constructed
         * objects
         * @param numSupercells number of supercells in the interval along each direction in the constructed objects
         */
        IntervalMapperFactory(DataSpace<T_dim> const& beginSupercell, DataSpace<T_dim> const& numSupercells)
            : beginSupercell(beginSupercell)
            , numSupercells(numSupercells)
        {
        }

        /** Construct an interval mapper object
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
            return IntervalMapping<T_MappingDescription>{mappingDescription, beginSupercell, numSupercells};
        }

        //! The first supercell index of the interval, including guards, in the constructed objects
        DataSpace<T_dim> const beginSupercell;

        //! Number of supercells in the interval along each direction in the constructed objects
        DataSpace<T_dim> const numSupercells;
    };

    /** Construct an interval mapper instance for the given description and parameters
     *
     * @tparam T_MappingDescription mapping description type
     * @tparam T_dim dimensionality of mapper
     *
     * @param mappingDescription mapping description
     * @param beginSupercell the first supercell index of the interval, including guards
     * @param numSupercells number of supercells in the interval along each direction
     */
    template<typename T_MappingDescription, unsigned T_dim>
    HINLINE auto makeIntervalMapper(
        T_MappingDescription mappingDescription,
        DataSpace<T_dim> const& beginSupercell,
        DataSpace<T_dim> const& numSupercells)
    {
        return IntervalMapperFactory<T_dim>{beginSupercell, numSupercells}(mappingDescription);
    }

} // namespace pmacc
