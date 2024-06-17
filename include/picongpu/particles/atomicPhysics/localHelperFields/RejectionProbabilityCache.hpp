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

// need: picongpu/param/atomicPhysics_Debug.param
#include "picongpu/simulation_defines.hpp"

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
    struct PrintRejectionProbabilityCacheToConsole
    {
        template<typename T_Acc, typename T_RejectionProbabilityCache>
        HDINLINE auto operator()(
            T_Acc const&,
            T_RejectionProbabilityCache const& rejectionProbabilityCache,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
            constexpr uint16_t numBins = T_RejectionProbabilityCache::numberBins;

            // check if overSubscribed
            bool overSubscription = false;
            for(uint16_t i = 0u; i < numBins; i++)
            {
                if(rejectionProbabilityCache.rejectionProbability(i) > 0._X)
                    overSubscription = true;
            }

            // print content
            std::cout << "rejectionProbabilityCache " << superCellIdx.toString(",", "[]");
            std::cout << " oversubcribed: " << ((overSubscription) ? "true" : "false") << std::endl;
            for(uint16_t i = 0u; i < numBins; i++)
            {
                if constexpr(printOnlyOversubscribed)
                {
                    if(rejectionProbabilityCache.rejectionProbability(i) < 0._X)
                        continue;
                }

                std::cout << "\t\t" << i << ":[ " << std::setw(10) << std::scientific
                          << rejectionProbabilityCache.rejectionProbability(i) << std::defaultfloat << " ]"
                          << std::endl;
            }
        }

        template<typename T_Acc, typename T_RejectionProbabilityCache>
        HDINLINE auto operator()(
            T_Acc const&,
            T_RejectionProbabilityCache const& rejectionProbabilityCache,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };

    /** @class cache of rejection probability p of over subscribed bin,
     *
     * p = (binDeltaWeight - binWeight0)/binDeltaWeight
     *
     * @tparam T_numberBins number of entries in cache
     *
     * @attention invalidated every time the local electron spectrum changes
     */
    template<uint32_t T_numberBins>
    class RejectionProbabilityCache
    {
    public:
        static constexpr uint32_t numberBins = T_numberBins;

    private:
        float_X rejectionProbabilities[T_numberBins] = {-1._X}; // unitless

    public:
        /** set cache entry
         *
         * @param binIndex
         * @param rejectionProbability rejectionProbability of bin
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         */
        HDINLINE void set(uint32_t const binIndex, float_X rejectionProbability)
        {
            if constexpr(picongpu::atomicPhysics::debug::rejectionProbabilityCache::BIN_INDEX_RANGE_CHECK)
                if(binIndex >= T_numberBins)
                {
                    printf("atomicPhysics ERROR: out of range in set() call on RejectionProbabilityCache\n");
                    return;
                }

            rejectionProbabilities[binIndex] = rejectionProbability;
            return;
        }

        /** get cached rate for an atomic state
         *
         * @param binIndex
         * @return rejectionProbability rejectionProbability of bin
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         */
        HDINLINE float_X rejectionProbability(uint32_t const binIndex) const
        {
            if constexpr(picongpu::atomicPhysics::debug::rejectionProbabilityCache::BIN_INDEX_RANGE_CHECK)
                if(binIndex >= T_numberBins)
                {
                    printf("atomicPhysics ERROR: out of range in rejectionProbability() call\n");
                    return -1._X;
                }

            return rejectionProbabilities[binIndex];
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
