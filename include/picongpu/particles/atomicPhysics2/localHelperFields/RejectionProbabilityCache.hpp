/* Copyright 2023 Brian Marre
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp" // need: picongpu/param/atomicPhysics2_Debug.param

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::localHelperFields
{
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
            if constexpr(picongpu::atomicPhysics2::debug::rejectionProbabilityCache::BIN_INDEX_RANGE_CHECK)
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
            if constexpr(picongpu::atomicPhysics2::debug::rejectionProbabilityCache::BIN_INDEX_RANGE_CHECK)
                if(binIndex >= T_numberBins)
                {
                    printf("atomicPhysics ERROR: out of range in rejectionProbability() call\n");
                    return -1._X;
                }

            return rejectionProbabilities[binIndex];
        }
    };
} // namespace picongpu::particles::atomicPhysics2::localHelperFields
