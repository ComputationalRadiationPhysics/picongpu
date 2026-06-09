/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2023-2024 Brian Marre
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>
#include <iomanip>
#include <iostream>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** debug only, write content of rate cache to console
     *
     * @attention only useful of compiling for serial backend, otherwise different RejectionProbabilityCache's outputs
     *  will interleave
     */
    template<bool printOnlyOversubscribed>
    struct PrintRejectionProbabilityCacheCellToConsole
    {
        template<typename T_Acc, typename T_RejectionProbabilityCache_Cell>
        HDINLINE auto operator()(
            T_Acc const&,
            T_RejectionProbabilityCache_Cell const& rejectionProbabilityCache,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
            constexpr uint32_t numCells = T_RejectionProbabilityCache_Cell::numberCells;

            // check if overSubscribed
            bool overSubscription = false;
            for(uint32_t i = 0u; i < numCells; i++)
            {
                if(rejectionProbabilityCache.getRejectionProbabilityCell(i) > 0._X)
                    overSubscription = true;
            }

            // print content
            std::cout << "rejectionProbabilityCache_Cell " << superCellIdx.toString(",", "[]");
            std::cout << " oversubcribed: " << ((overSubscription) ? "true" : "false") << std::endl;
            std::cout << "Cells:" << std::endl;
            for(uint32_t i = 0u; i < numCells; i++)
            {
                if constexpr(printOnlyOversubscribed)
                {
                    if(rejectionProbabilityCache.getRejectionProbabilityCell(i) < 0._X)
                        continue;
                }

                std::cout << "\t\t" << i << ":[ " << std::setw(10) << std::scientific
                          << rejectionProbabilityCache.getRejectionProbabilityCell(i) << std::defaultfloat << " ]"
                          << std::endl;
            }
        }

        template<typename T_Acc, typename T_RejectionProbabilityCache_Cell>
        HDINLINE auto operator()(
            T_Acc const&,
            T_RejectionProbabilityCache_Cell const& rejectionProbabilityCache,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };

    /** cache of rejection probability of p_cell for over subscribed cells,
     *
     * p_Cell = (cellEnergy - cellEnergyUsed)/cellDeltaEnergy
     *
     * @tparam T_numberCells number of cell entries in cache
     *
     * @attention invalidated every time the FieldEnergyUse changes
     */
    template<uint32_t T_numberCells>
    class RejectionProbabilityCache_Cell
    {
    public:
        static constexpr uint32_t numberCells = T_numberCells;

    private:
        float_X rejectionProbabilityCell[numberCells] = {-1._X}; // unitless

        //! @attention only active by debug setting
        HDINLINE static bool outOfRangeLinearCellIndex(uint32_t const linearCellIndex)
        {
            if constexpr(picongpu::atomicPhysics::debug::rejectionProbabilityCache::BIN_INDEX_RANGE_CHECK)
                if(linearCellIndex >= numberCells)
                {
                    printf(
                        "atomicPhysics ERROR: out of range linear cell index in call to "
                        "RejectionProbabilityCache_Cell\n");
                    return true;
                }
            return false;
        }

    public:
        /** set cache entry
         *
         * @param linearCellIndex linearized index of cell
         * @param rejectionProbability rejectionProbability of cell
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         */
        HDINLINE void setCell(uint32_t const linearCellIndex, float_X const rejectionProbability)
        {
            if(outOfRangeLinearCellIndex(linearCellIndex))
                return;

            rejectionProbabilityCell[linearCellIndex] = rejectionProbability;
        }

        /** get cached rejectionProbability for a cell
         *
         * @param linearCellIndex
         * @return rejectionProbability rejectionProbability of bin
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         */
        HDINLINE float_X getRejectionProbabilityCell(uint32_t const linearCellIndex) const
        {
            if(outOfRangeLinearCellIndex(linearCellIndex))
                return -1._X;

            return rejectionProbabilityCell[linearCellIndex];
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
