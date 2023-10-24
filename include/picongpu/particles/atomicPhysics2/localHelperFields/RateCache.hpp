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
#include "picongpu/particles/atomicPhysics2/enums/TransitionType.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionDirection.hpp"
#include "picongpu/particles/atomicPhysics2/enums/transitionDataSet.hpp"
#include "picongpu/particles/atomicPhysics2/ConvertEnumToUint.hpp"

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
        static constexpr uint32_t numberDataSets = atomicPhyiscs2::enums::numberTransitionDataSets-1u;
    private:
        // partial sums of rates for each atomic state, one for each TransitionDataSet except noChange
        // 1/UNIT_TIME
        float_X rateEntries[T_numberAtomicStates * numberDataSets] = {0};
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
                if((collectionIndex >= numberAtomicStates) || (dataSetIndex >= numberDataSets))
                {
                    printf("atomciPhysics ERROR: out of range linearIndex() call to rateCache\n")
                    return ;
                }
            }
            return numberDataSets * collectionIndex + dataSetIndex;
        }

    public:
        /** add to cache entry, using atomics
         *
         * @tparam T_TransitionDataSet TransitionDataSet to add rate to
         *
         * @param worker object containing the device and block information
         * @param collectionIndex collection Index of atomic state
         * @param rate rate of transition, [1/UNIT_TIME]
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<
            atomicPhysics2::enums::TransitionDataSet T_TransitionDataSet,
            typename T_Worker>
        HDINLINE void add(T_Worker const& worker, uint32_t const collectionIndex, float_X rate)
        {

            PMACC_CASSERT_MSG(
                noChange_not_allowed_as_T_TransitionDataSet,
                u32(T_TransitionDataSet) == numberDataSets)
            PMACC_CASSERT_MSG(
                unknown_T_TransitionDataSet,
                u32(T_TransitionDataSet) > numberDataSets)

            constexpr uint32_t offset = offset<T_TransitionDataSet>();

            cupla::atomicAdd(
                worker.getAcc(),
                &(this->rateEntries[linearIndex(collectionIndex, u32(T_TransitionDataSet))]),
                rate);
            return;
        }

        /** add to cache entry, no atomics
         *
         * @tparam T_TransitionDataSet TransitionDataSet to add rate to
         *
         * @param collectionIndex collection Index of atomic state
         * @param rate rate of transition, [1/UNIT_TIME]
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         */
        template<atomicPhysics2::enums::TransitionDataSet T_TransitionDataSet>
        HDINLINE void add(uint32_t const collectionIndex, float_X rate)
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in add() call on RateCache\n");
                    return;
                }

           PMACC_CASSERT_MSG(
                noChange_not_allowed_as_T_TransitionDataSet,
                u32(T_TransitionDataSet) == numberDataSets)
            PMACC_CASSERT_MSG(
                unknown_T_TransitionDataSet,
                u32(T_TransitionDataSet) > numberDataSets)

            rateEntries[linearIndex(collectionIndex, u32(T_TransitionDataSet))]
                = rateEntries[linearIndex(collectionIndex, u32(T_TransitionDataSet))] + rate;
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
        HDINLINE void setPresent(T_Worker const& worker, uint32_t const collectionIndex, bool const status)
        {
            cupla::atomicExch(
                worker.getAcc(),
                &(this->m_present[linearIndex(collectionIndex, u32(T_TransitionDataSet))]),
                static_cast<uint32_t>(status));
            return;
        }

        /** get cached rate for an atomic state
         *
         * @param transitionDataSetIndex colelctionIndex of transitionDataSet
         * @param collectionIndex collection index of atomic state
         * @return rate of transition, [1/UNIT_TIME], 0 if unknown T_TransitionType T_TransitionDirection combination
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         * @attention returns invalid value if state not present
         */
        template<>
        HDINLINE float_X rate(uint8_t const transitionDataSetIndex, uint32_t const collectionIndex) const
        {
           PMACC_CASSERT_MSG(
                noChange_not_allowed_as_T_TransitionDataSet,
                u32(T_TransitionDataSet) == numberDataSets)
            PMACC_CASSERT_MSG(
                unknown_T_TransitionDataSet,
                u32(T_TransitionDataSet) > numberDataSets)

            return rateEntries[linearIndex(collectionIndex, u32(T_TransitionDataSet))];
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
            for (uint8_t i=0u; i < numberDataSets; ++i)
            {
                totalLossRate += rateEntries[linearIndex(collectionIndex, u32(T_TransitionDataSet))]
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
                    printf("atomicPhysics ERROR: out of range in rate() call on rateCache\n");
                    return 0._X;
                }

            return m_present[collectionIndex];
        }

        //! debug only, write content of rate cache to console, @attention serial and cpu build only
        HINLINE void printToConsole(pmacc::DataSpace<picongpu::simDim> superCellFieldIdx) const
        {
            std::cout << "rateCache" << superCellFieldIdx.toString(",", "[]")
                << " atomicStateCollectionIndex [bb(up), bb(down), bf(up), a(down)]" << std::endl;
            for(uint16_t i = 0u; i < numberAtomicStates; i++)
            {
                if(this->present(i))
                {
                    std::cout << "\t" << i << "[";
                    for (uint8_t i=0u; i < (numberDataSets-1u); ++i)
                    {
                        std::cout << rateEntries[linearIndex(collectionIndex, i)] << ", ";
                    }
                    // last dataSet
                    std::cout << rateEntries[linearIndex(collectionIndex, numberDataSets-1u)] << "]" << std::endl;
                }
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics2::localHelperFields
