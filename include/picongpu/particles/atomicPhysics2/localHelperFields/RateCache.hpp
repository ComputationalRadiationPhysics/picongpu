/* Copyright 2022-2023 Brian Marre
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
    /** @class cache of one row, column, or the diagonal of the rate matrix
     *
     * @tparam T_numberAtomicStates number of entries in cache
     *
     * @attention invalidated every time the local electron spectrum changes
     */
    template<uint32_t T_numberAtomicStates>
    class RateCache
    {
    public:
        static constexpr uint32_t numberAtomicStates = T_numberAtomicStates;

    private:
        float_X rates[T_numberAtomicStates] = {0}; // unit: 1/(Dt_PIC)
        bool m_present[T_numberAtomicStates] = {false}; // unitless

    public:
        /** add to cache entry, using atomics
         *
         * @param worker object containing the device and block information
         * @param collectionIndex collection Index of atomic state
         * @param rate rate of transition, [1/UNIT_TIME]
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<typename T_Worker>
        HDINLINE void add(T_Worker const& worker, uint32_t const collectionIndex, float_X rate)
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in add() call on RateCache\n");
                    return;
                }

            cupla::atomicAdd(worker.getAcc(), &(this->rates[collectionIndex]), rate);
            return;
        }

        /** add to cache entry, no atomics
         *
         * @param collectionIndex collection Index of atomic state
         * @param rate rate of transition, [1/UNIT_TIME]
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         */
        HDINLINE void add(uint32_t const collectionIndex, float_X rate)
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in add() call on RateCache\n");
                    return;
                }

            rates[collectionIndex] = rates[collectionIndex] + rate;
            return;
        }

        /** set indicator if atomic state is present
         *
         * @param worker object containing the device and block information
         * @param collectionIndex collection Index of atomic state
         * @param status presence status to set, true =^= present, false =^= not present
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<typename T_Worker>
        HDINLINE void setPresent(T_Worker const& worker, uint32_t const collectionIndex, bool const status)
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in add() call on RateCache\n");
                    return;
                }

            cupla::atomicExch(worker.getAcc(), &(this->m_present[collectionIndex]), status);
            return;
        }

        /** get cached rate for an atomic state
         *
         * @param collectionIndex collection Index of atomic state
         * @return rate of transition, [1/UNIT_TIME]
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         */
        HDINLINE float_X rate(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in rate() call on rateCache\n");
                    return 0._X;
                }

            return rates[collectionIndex];
        }

        /** get presence status for an atomic state
         *
         * @param collectionIndex collection Index of atomic state
         * @return if state is present in superCell
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         */
        HDINLINE bool present(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in rate() call on rateCache\n");
                    return 0._X;
                }

            return m_present[collectionIndex];
        }
    };
} // namespace picongpu::particles::atomicPhysics2::localHelperFields
