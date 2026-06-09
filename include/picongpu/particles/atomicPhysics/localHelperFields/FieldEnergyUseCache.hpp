/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2024-2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

// need simDim from dimension.param and SuperCellSize from memory.param
#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"

#include <pmacc/attribute/unroll.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** cache of the field energy use of a block of cells described by T_Extent, typically for all cells of a superCell
     *
     * @param T_Extent pmacc compile time vector describing extent of cache in number of cell
     * @param T_StorageType type to use for storage
     */
    template<typename T_Extent, typename T_StorageType>
    struct FieldEnergyUseCache
    {
        using Extent = T_Extent;
        using StorageType = T_StorageType;

        static constexpr uint32_t dim = Extent::dim;
        static constexpr uint32_t numberCells = pmacc::math::CT::volume<Extent>::type::value;

        using CellIdx = pmacc::DataSpace<dim>;

    private:
        // eV
        StorageType fieldEnergyUsed[numberCells] = {0._X};

        //! @returns passes silently if ok
        HDINLINE static void checkWithinExtent(CellIdx const& extent, CellIdx const& cellIdx)
        {
            if constexpr(picongpu::atomicPhysics::debug::fieldEnergyUseCache::CELL_INDEX_RANGE_CHECKS)
            {
                constexpr uint32_t iExtent = dim;
                PMACC_UNROLL(iExtent)
                for(uint8_t i = 0u; i < dim; ++i)
                {
                    if(cellIdx[i] >= extent[i])
                    {
                        printf(
                            "atomicPhysics ERROR: out of range in cellIndex based call to FieldEnergyUsedCachein\n");
                    }
                }
            }
        }

        //! @returns passes silently if ok
        HDINLINE static void checkWithinLinearExtent(uint32_t const linearCellIndex)
        {
            if constexpr(picongpu::atomicPhysics::debug::fieldEnergyUseCache::CELL_INDEX_RANGE_CHECKS)
            {
                if(linearCellIndex >= numberCells)
                {
                    printf(
                        "atomicPhysics ERROR: out of range in linearCellIndex based call to FieldEnergyUsedCachein\n");
                }
            }
        }

    public:
        /** add to cache entry using atomics
         *
         * @param worker object containing the device and block information
         * @param localCellIndex vector index of cell to add energyUsed to
         * @param energy energy used, in eV
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<typename T_Worker>
        HDINLINE void add(T_Worker const& worker, CellIdx const localCellIndex, float_X const energyUsed)
        {
            checkWithinExtent(Extent::toRT(), localCellIndex);

            alpaka::atomicAdd(
                worker.getAcc(),
                &(this->fieldEnergyUsed[pmacc::math::linearize(Extent::toRT(), localCellIndex)]),
                energyUsed,
                ::alpaka::hierarchy::Threads{});
        }

        /** add to cache entry using atomics and direct access version
         *
         * @param worker object containing the device and block information
         * @param linearCellIndex linear index of cell
         * @param energy energy used, in eV
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<typename T_Worker>
        HDINLINE void add(T_Worker const& worker, uint32_t const linearCellIndex, float_X const energyUsed)
        {
            checkWithinLinearExtent(linearCellIndex);

            alpaka::atomicAdd(
                worker.getAcc(),
                &(this->fieldEnergyUsed[linearCellIndex]),
                energyUsed,
                ::alpaka::hierarchy::Threads{});
        }

        /** add to cache entry, no atomics
         *
         * @param localCellIndex vector index of cell to add energyUsed to
         * @param energyUsed energy used, in eV, weighted
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         */
        HDINLINE void add(CellIdx const localCellIndex, float_X const energyUsed)
        {
            checkWithinExtent(Extent::toRT(), localCellIndex);

            fieldEnergyUsed[pmacc::math::linearize(Extent::toRT(), localCellIndex)] += energyUsed;
        }

        /** add to cache entry, no atomics and direct access version
         *
         * @param linearCellIndex linear index of cell
         * @param energyUsed energy used, in eV, weighted
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         * @attention direct access to 1D data storage, check data layout before use!
         */
        HDINLINE void add(uint32_t const linearCellIndex, float_X const energyUsed)
        {
            checkWithinLinearExtent(linearCellIndex);

            fieldEnergyUsed[linearCellIndex] += energyUsed;
        }

        /** get used field energy of cell
         *
         * @param localCellIndex vector index of cell to add energyUsed to
         *
         * @attention no range checks outside a debug compile, invalid memory read on failure
         *
         * @return unit: eV
         */
        HDINLINE StorageType energyUsed(CellIdx const localCellIndex) const
        {
            checkWithinExtent(Extent::toRT(), localCellIndex);

            return fieldEnergyUsed[pmacc::math::linearize(Extent::toRT(), localCellIndex)];
        }

        /** get used field energy of cell, direct access version
         *
         * @param linearCellIndex linear index of cell
         *
         * @attention no range checks outside a debug compile, invalid memory read on failure
         * @attention direct access to 1D data storage, check data layout before use!
         *
         * @return unit: eV
         */
        HDINLINE StorageType energyUsed(uint32_t const linearCellIndex) const
        {
            checkWithinLinearExtent(linearCellIndex);

            return fieldEnergyUsed[linearCellIndex];
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
