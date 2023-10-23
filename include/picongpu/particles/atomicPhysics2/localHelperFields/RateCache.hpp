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
#include "picongpu/particles/atomicPhysics2/ConvertEnumToUint.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

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

    private:
        // partial sums of rates for each atomic state
        // 1/UNIT_TIME
        float_X rateBoundBoundUpward[T_numberAtomicStates] = {0};
        // 1/UNIT_TIME
        float_X rateBoundBoundDownward[T_numberAtomicStates] = {0};
        // 1/UNIT_TIME
        float_X rateBoundFreeUpward[T_numberAtomicStates] = {0};
        // 1/UNIT_TIME
        float_X rateAutonomousDownward[T_numberAtomicStates] = {0};
        /// @todo add rate_boundFree_Downward for recombination, Brian Marre, 2023

        uint32_t m_present[T_numberAtomicStates] = {static_cast<uint32_t>(false)}; // unitless

    public:
        /** add to cache entry, using atomics
         *
         * @tparam T_TransitionType type of transition
         * @tparam T_TransitionDirection direction of transition
         *
         * @param worker object containing the device and block information
         * @param collectionIndex collection Index of atomic state
         * @param rate rate of transition, [1/UNIT_TIME]
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<
            atomicPhysics2::enums::TransitionType T_TransitionType,
            atomicPhysics2::enums::TransitionDirection T_TransitionDirection,
            typename T_Worker>
        HDINLINE void add(T_Worker const& worker, uint32_t const collectionIndex, float_X rate)
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in add() call on RateCache\n");
                    return;
                }

            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::boundBound))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::upward)))
            {
                cupla::atomicAdd(worker.getAcc(), &(this->rateBoundBoundUpward[collectionIndex]), rate);
            }
            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::boundBound))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::downward)))
            {
                cupla::atomicAdd(worker.getAcc(), &(this->rateBoundBoundDownward[collectionIndex]), rate);
            }
            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::boundFree))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::upward)))
            {
                cupla::atomicAdd(worker.getAcc(), &(this->rateBoundFreeUpward[collectionIndex]), rate);
            }
            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::autonomous))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::downward)))
            {
                cupla::atomicAdd(worker.getAcc(), &(this->rateAutonomousDownward[collectionIndex]), rate);
            }

            // for unknown do nothing

            return;
        }

        /** add to cache entry, no atomics
         *
         * @tparam T_TransitionType type of transition
         * @tparam T_TransitionDirection direction of transition
         *
         * @param collectionIndex collection Index of atomic state
         * @param rate rate of transition, [1/UNIT_TIME]
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         */
        template<
            atomicPhysics2::enums::TransitionType T_TransitionType,
            atomicPhysics2::enums::TransitionDirection T_TransitionDirection>
        HDINLINE void add(uint32_t const collectionIndex, float_X rate)
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in add() call on RateCache\n");
                    return;
                }

            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::boundBound))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::upward)))
            {
                rateBoundBoundUpward[collectionIndex] = rateBoundBoundUpward[collectionIndex] + rate;
            }
            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::boundBound))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::downward)))
            {
                rateBoundBoundDownward[collectionIndex] = rateBoundBoundDownward[collectionIndex] + rate;
            }
            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::boundFree))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::upward)))
            {
                rateBoundFreeUpward[collectionIndex] = rateBoundFreeUpward[collectionIndex] + rate;
            }
            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::autonomous))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::downward)))
            {
                rateBoundFreeUpward[collectionIndex] = rateAutonomousDownward[collectionIndex] + rate;
            }

            // for unknown do nothing

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
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in add() call on RateCache\n");
                    return;
                }

            cupla::atomicExch(worker.getAcc(), &(this->m_present[collectionIndex]), static_cast<uint32_t>(status));
            return;
        }

        /** get cached rate for an atomic state
         *
         * @tparam T_TransitionType type of transition
         * @tparam T_TransitionDirection direction of transition
         *
         * @param collectionIndex collection Index of atomic state
         * @return rate of transition, [1/UNIT_TIME], 0 if unknown T_TransitionType T_TransitionDirection combination
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         * @attention returns invalid value if state not present
         */
        template<
            atomicPhysics2::enums::TransitionType T_TransitionType,
            atomicPhysics2::enums::TransitionDirection T_TransitionDirection>
        HDINLINE float_X rate(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::rateCache::COLLECTION_INDEX_RANGE_CHECKS)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in rate() call on rateCache\n");
                    return 0._X;
                }

            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::boundBound))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::upward)))
            {
                return rateBoundBoundUpward[collectionIndex];
            }
            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::boundBound))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::downward)))
            {
                return rateBoundBoundDownward[collectionIndex];
            }
            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::boundFree))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::upward)))
            {
                return rateBoundFreeUpward[collectionIndex];
            }
            if constexpr(
                (u8(T_TransitionType) == u8(atomicPhyiscs2::enums::TransitionType::autonomous))
                && (u8(T_TransitionDirection) == u8(atomicPhyiscs2::enums::TransitionDirection::downward)))
            {
                return rateBoundFreeUpward[collectionIndex];
            }

            // unknown combination of T_TransitionType and T_TransitionDirection
            return 0._X;
        }

        /** get chached total loss rate for an atomic state
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
            return rateBoundBoundUpward[collectionIndex]
                + rateBoundBoundDownward[collectionIndex]
                + rateBoundFreeUpward[collectionIndex]
                + rateAutonomousDownward[collectionIndex];
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
                    std::cout << "\t" << i << "["
                        << this->rateBoundBoundUpward[i] << ", "
                        << this->rateBoundBoundDownward[i] << ", "
                        << this->rateBoundFreeUpward[i] << ", "
                        << this->rateAutonomousDownward[i] << ", "
                        << "]" << std::endl;
                }
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics2::localHelperFields
