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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
// need: picongpu/param/atomicPhysics2_Debug.param

#include "picongpu/particles/atomicPhysics2/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionDataSet.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionDirection.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionType.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/static_assert.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::localHelperFields
{
    /** cache of accumulated rates of each atomic state for each transition type and ordering
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
        // we do not store noChange since noChange is always reminder to 1
        static constexpr uint32_t numberStoredDataSets
            = particles::atomicPhysics2::enums::numberTransitionDataSets - 1u;

    private:
        // partial sums of rates for each atomic state, one for each TransitionDataSet except noChange
        // 1/UNIT_TIME
        float_X rateEntries[T_numberAtomicStates * numberStoredDataSets] = {0};
        uint32_t m_present[T_numberAtomicStates] = {static_cast<uint32_t>(false)}; // unitless

        /** get linear storage index
         *
         * @param collectionIndex atomic state collection index of an atomic state
         * @param dataSetIndex index of data set
         */
        HDINLINE static constexpr uint32_t linearIndex(uint32_t const collectionIndex, uint32_t const dataSetIndex)
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
            {
                if((collectionIndex >= numberAtomicStates) || (dataSetIndex >= numberStoredDataSets))
                {
                    printf("atomciPhysics ERROR: out of range linearIndex() call to rateCache\n");
                    return 0u;
                }
            }
            return numberStoredDataSets * collectionIndex + dataSetIndex;
        }

    public:
        /** add to cache entry, using atomics
         *
         * @param worker object containing the device and block information
         * @param collectionIndex collection Index of atomic state to add rate to
         * @param rate rate of transition, [1/UNIT_TIME]
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<typename T_Worker, particles::atomicPhysics2::enums::TransitionDataSet T_TransitionDataSet>
        HDINLINE void add(T_Worker const& worker, uint32_t const collectionIndex, float_X rate)
        {
            PMACC_CASSERT_MSG(
                noChange_not_allowed_as_T_TransitionDataSet,
                u32(T_TransitionDataSet) != u32(particles::atomicPhysics2::enums::TransitionDataSet::noChange));
            PMACC_CASSERT_MSG(
                unknown_T_TransitionDataSet,
                u32(T_TransitionDataSet) < particles::atomicPhysics2::enums::numberTransitionDataSets);

            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in add() call on RateCache\n");
                    return;
                }

            alpaka::atomicAdd(
                worker.getAcc(),
                &(this->rateEntries[linearIndex(collectionIndex, u32(T_TransitionDataSet))]),
                rate,
                ::alpaka::hierarchy::Threads{});
        }

        /** add to cache entry, no atomics
         *
         * @tparam T_TransitionDataSet TransitionDataSet to add rate to
         *
         * @param collectionIndex collection Index of atomic state to add rate to
         * @param rate rate of transition, [1/UNIT_TIME]
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         */
        template<particles::atomicPhysics2::enums::TransitionDataSet T_TransitionDataSet>
        HDINLINE void add(uint32_t const collectionIndex, float_X rate)
        {
            PMACC_CASSERT_MSG(
                noChange_not_allowed_as_T_TransitionDataSet,
                u32(T_TransitionDataSet) != u32(particles::atomicPhysics2::enums::TransitionDataSet::noChange));
            PMACC_CASSERT_MSG(
                unknown_T_TransitionDataSet,
                u32(T_TransitionDataSet) < particles::atomicPhysics2::enums::numberTransitionDataSets);

            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in add() call on RateCache\n");
                    return;
                }

            rateEntries[linearIndex(collectionIndex, u32(T_TransitionDataSet))] += rate;
            return;
        }

        /** set indicator if atomic state is present
         *
         * @param worker object containing the device and block information
         * @param collectionIndex collection index of atomic state to set present for
         * @param status presence status to set, true =^= present, false =^= not present
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<typename T_Worker>
        HDINLINE void setPresent(T_Worker const& worker, uint32_t const collectionIndex, bool const status)
        {
            alpaka::atomicExch(
                worker.getAcc(),
                &(this->m_present[collectionIndex]),
                static_cast<uint32_t>(status),
                ::alpaka::hierarchy::Threads{});
            return;
        }

        /** get cached rate for an atomic state
         *
         * @param transitionDataSetIndex collectionIndex of transitionDataSet to get rate for
         * @param collectionIndex collection index of atomic state to get rate for
         * @return rate of transition, [1/UNIT_TIME], 0 if unknown T_TransitionType T_TransitionDirection combination
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         * @attention returns invalid value if state not present
         */
        HDINLINE float_X rate(uint32_t const transitionDataSetIndex, uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::TRANSITION_DATA_SET_INDEX_RANGE_CHECKS)
            {
                if(transitionDataSetIndex
                   == u8(picongpu::particles::atomicPhysics2::enums::TransitionDataSet::noChange))
                {
                    printf("atomicPhysics ERROR: noChange not allowed as transitionDataSet in rate() call\n");
                    return 0._X;
                }
                if(transitionDataSetIndex >= u8(picongpu::particles::atomicPhysics2::enums::numberTransitionDataSets))
                {
                    printf("atomicPhysics ERROR: unknown transitionDataSet index in rate() call\n");
                    return 0._X;
                }
            }

            return rateEntries[linearIndex(collectionIndex, transitionDataSetIndex)];
        }

        /** get cached total loss rate for an atomic state
         *
         * @param collectionIndex collection Index of atomic state
         * @return rate of transition, [1/UNIT_TIME], by convention >0
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         * @attention returns invalid value if state not present
         */
        HDINLINE float_X totalLossRate(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in totalLossRate() call on rateCache\n");
                    return 0._X;
                }

            float_X totalLossRate = 0._X;
            for(uint32_t transitionDataSetIndex = 0u; transitionDataSetIndex < numberStoredDataSets;
                ++transitionDataSetIndex)
            {
                totalLossRate += rateEntries[linearIndex(collectionIndex, transitionDataSetIndex)];
            }
            return totalLossRate;
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
                    printf("atomicPhysics ERROR: out of range in present() call on rateCache\n");
                    return false;
                }

            return m_present[collectionIndex];
        }

        //! debug only, write content of rate cache to console, @attention serial and cpu build only
        HINLINE void printToConsole(pmacc::DataSpace<picongpu::simDim> superCellFieldIdx) const
        {
            std::cout << "rateCache" << superCellFieldIdx.toString(",", "[]")
                      << " atomicStateCollectionIndex [bb(up), bb(down), bf(up), a(down)]" << std::endl;
            for(uint16_t collectionIndex = 0u; collectionIndex < numberAtomicStates; ++collectionIndex)
            {
                if(this->present(collectionIndex))
                {
                    std::cout << "\t" << collectionIndex << "[";
                    for(uint32_t transitionDataSetIndex = 0u; transitionDataSetIndex < (numberStoredDataSets - 1u);
                        ++transitionDataSetIndex)
                    {
                        std::cout << rateEntries[linearIndex(collectionIndex, transitionDataSetIndex)] << ", ";
                    }
                    // last dataSet
                    std::cout << rateEntries[linearIndex(collectionIndex, numberStoredDataSets - 1u)] << "]"
                              << std::endl;
                }
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics2::localHelperFields
