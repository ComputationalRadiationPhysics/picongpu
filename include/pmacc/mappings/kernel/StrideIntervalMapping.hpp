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
    /** Strided mapping from block indices to supercells in the given interval for alpaka kernels
     *
     * An interval is a T_dim-dimensional contiguous Cartesian range.
     * Parameters of the inverval are defined at runtime, unlike most other mappers.
     *
     * Adheres to the MapperConcept.
     *
     * The mapped interval is subdivided into stride^dim non-intersecting subintervals (some may be empty).
     * A subinterval is an intersection of the interval and an integer lattice with given stride in all directions.
     * Each subinterval has a unique offset relative to interval start.
     *
     * @tparam T_MappingDescription mapping description type
     * @tparam T_stride stride value, same for all directions
     */
    template<typename T_MappingDescription, uint32_t T_stride>
    class StrideIntervalMapping;

    template<
        template<unsigned, class>
        class T_MappingDescription,
        unsigned T_dim,
        typename T_SuperCellSize,
        uint32_t T_stride>
    class StrideIntervalMapping<T_MappingDescription<T_dim, T_SuperCellSize>, T_stride>
        : public T_MappingDescription<T_dim, T_SuperCellSize>
    {
    public:
        //! Base class
        using BaseClass = T_MappingDescription<T_dim, T_SuperCellSize>;

        //! Compile-time super cell size
        using SuperCellSize = typename BaseClass::SuperCellSize;

        //! Dimensionality value
        static constexpr uint32_t dim = T_dim;

        //! Stride value
        static constexpr uint32_t stride = T_stride;

        /** Create a mapper instance
         *
         * @param base instance of the base class to be propagated
         * @param beginSupercell the first supercell index of the interval, including guards
         * @param numSupercells number of supercells in the interval along each direction
         */
        HINLINE StrideIntervalMapping(
            BaseClass base,
            DataSpace<dim> const& beginSupercell,
            DataSpace<dim> const& numSupercells)
            : BaseClass(base)
            , beginSupercell(beginSupercell)
            , numSupercells(numSupercells)
            , offset(DataSpace<dim>::create(0))
        {
        }

        StrideIntervalMapping(const StrideIntervalMapping&) = default;

        /** Generate grid dimension information for alpaka kernel calls
         *
         * A kernel using this mapping must use exacly the returned number of blocks
         *
         * @return number of blocks in a grid
         */
        HINLINE DataSpace<dim> getGridDim() const
        {
            return (numSupercells - offset + stride - 1) / stride;
        }

        /** Return index of a supercell to be processed by the given alpaka block
         *
         * @param blockIdx alpaka block index
         * @return mapped SuperCell index including guards
         */
        HDINLINE DataSpace<dim> getSuperCellIndex(DataSpace<dim> const& blockIdx) const
        {
            return beginSupercell + offset + blockIdx * stride;
        }

        //! Get current offset value
        HDINLINE DataSpace<dim> getOffset() const
        {
            return offset;
        }

        /** Set a new offset value
         *
         * @param newOffset new offset value
         */
        HDINLINE void setOffset(DataSpace<dim> const newOffset)
        {
            offset = newOffset;
        }

        /** Set mapper to next non-empty subinterval
         *
         * Note: this function has no HINLINE as it is recursive and so cannot be force-inlined.
         *
         * @return whether the whole interval was processed
         */
        bool next()
        {
            int linearOffset = pmacc::math::linearize(DataSpace<dim>::create(stride), offset);
            linearOffset++;
            offset = pmacc::math::mapToND(DataSpace<dim>::create(stride), linearOffset);
            /* First check if everything is processed to have a recursion stop condition.
             * Then if the new grid dim has 0 size, immediately go to the next state.
             * This way it guarantees a valid grid dim after next() returns true.
             */
            if(linearOffset >= DataSpace<dim>::create(stride).productOfComponents())
                return false;
            if(getGridDim().productOfComponents() == 0)
                return next();
            return true;
        }

    private:
        //! The first supercell index of the interval, including guards
        DataSpace<T_dim> const beginSupercell;

        //! Number of supercells in the interval along each direction
        DataSpace<T_dim> const numSupercells;

        //! Offset of the current stride relative to beginSupercell
        DataSpace<T_dim> offset;
    };

    /** Construct a strided interval mapper instance for the given description
     *
     * Adheres to the MapperFactoryConcept.
     *
     * @tparam T_dim dimensionality of mappers to be constructed
     * @tparam T_stride stride value, same for all directions
     */
    template<unsigned T_dim, uint32_t T_stride>
    struct StrideIntervalMapperFactory
    {
        /** Create a factory instance
         *
         * @param beginSupercell the first supercell index of the interval, including guards, in the constructed
         * objects
         * @param numSupercells number of supercells in the interval along each direction in the constructed objects
         */
        StrideIntervalMapperFactory(DataSpace<T_dim> const& beginSupercell, DataSpace<T_dim> const& numSupercells)
            : beginSupercell(beginSupercell)
            , numSupercells(numSupercells)
        {
        }

        /** Construct a strided interval mapper object
         *
         * @tparam T_MappingDescription mapping description type
         *
         * @param mappingDescription mapping description
         *
         * @return an object adhering to the MapperConcept
         */
        template<typename T_MappingDescription>
        HINLINE auto operator()(T_MappingDescription mappingDescription) const
        {
            return StrideIntervalMapping<T_MappingDescription, T_stride>{
                mappingDescription,
                beginSupercell,
                numSupercells};
        }

        //! The first supercell index of the interval, including guards, in the constructed objects
        DataSpace<T_dim> const beginSupercell;

        //! Number of supercells in the interval along each direction in the constructed objects
        DataSpace<T_dim> const numSupercells;
    };

    /** Construct a strided interval mapper instance for the given description and parameters
     *
     * @tparam T_stride stride value, same for all directions
     * @tparam T_MappingDescription mapping description type
     * @tparam T_dim dimensionality of mapper
     *
     * @param mappingDescription mapping description
     * @param beginSupercell the first supercell index of the interval, including guards
     * @param numSupercells number of supercells in the interval along each direction
     */
    template<uint32_t T_stride, typename T_MappingDescription, unsigned T_dim>
    HINLINE auto makeStrideIntervalMapper(
        T_MappingDescription mappingDescription,
        DataSpace<T_dim> const& beginSupercell,
        DataSpace<T_dim> const& numSupercells)
    {
        return StrideIntervalMapperFactory<T_dim, T_stride>{beginSupercell, numSupercells}(mappingDescription);
    }

} // namespace pmacc
