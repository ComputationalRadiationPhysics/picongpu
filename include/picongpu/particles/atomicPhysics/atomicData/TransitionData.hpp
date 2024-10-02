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

#include "picongpu/particles/atomicPhysics/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/DataBuffer.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"

#include <cstdint>
#include <memory>
#include <string>

/** @file implements base class of transitions property data
 *
 * @attention ConfigNumber specifies the number of a state as defined by the configNumber
 *      class, while index always refers to a collection index.
 *      The configNumber of a given state is always the same, its collection index depends
 *      on input file,it should therefore only be used internally!
 */

namespace picongpu::particles::atomicPhysics::atomicData
{
    template<typename T_Number, typename T_Value, typename T_CollectionIndex>
    class TransitionDataBox : public DataBox<T_Number, T_Value>
    {
    public:
        using Idx = T_CollectionIndex;
        using BoxCollectionIndex = pmacc::DataBox<pmacc::PitchedBox<T_CollectionIndex, 1u>>;
        using S_DataBox = DataBox<T_Number, T_Value>;

    protected:
        //! configNumber of the lower(lower excitation energy) state of the transition
        BoxCollectionIndex m_boxLowerStateCollectionIndex;
        //! configNumber of the upper(higher excitation energy) state of the transition
        BoxCollectionIndex m_boxUpperStateCollectionIndex;

        uint32_t m_numberTransitions;
        // not constexpr, since only known at load input files, not const since needs to be trivially copyable

    public:
        /** constructor
         *
         * @attention transition data must be sorted block-wise ascending by lower/upper
         *  atomic state and secondary ascending by upper/lower atomic state.
         * @param boxLowerStateCollectionIndex collection index of the lower(lower excitation energy)
         *      state of the transition in a atomicState dataBox
         * @param boxUpperStateCollectionIndex collection index of the upper(higher excitation energy)
         *      state of the transition in a atomicState dataBox
         * @param numberTransitions number of transitions total in this box
         */
        TransitionDataBox(
            BoxCollectionIndex boxLowerStateCollectionIndex,
            BoxCollectionIndex boxUpperStateCollectionIndex,
            uint32_t numberTransitions)
            : m_boxLowerStateCollectionIndex(boxLowerStateCollectionIndex)
            , m_boxUpperStateCollectionIndex(boxUpperStateCollectionIndex)
            , m_numberTransitions(numberTransitions)
        {
        }

    protected:
        /** store transition in data box
         *
         * @attention do not forget to call syncToDevice() on the
         *  corresponding buffer, or the state is only added on the host side.
         * @attention needs to fulfill all ordering and content assumptions of constructor!
         * @attention no range check outside debug, collectionIndex >= numberTransitions will lead to invalid memory
         * access!
         *
         * @param collectionIndex index of data box entry to rewrite
         * @param lowerStateCollectionIndex collection index of lower state of transition in a atomic state dataBox
         * @param upperStateCollectionIndex collection index of upper state of transition in a atomic state dataBox
         */
        HINLINE void storeTransition(
            uint32_t const transitionCollectionIndex,
            Idx lowerStateCollectionIndex,
            Idx upperStateCollectionIndex)
        {
            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_LOAD)
                if(transitionCollectionIndex >= m_numberTransitions)
                {
                    throw std::runtime_error(
                        "atomicPhysics ERROR: out of range storeTransition("
                        + std::to_string(transitionCollectionIndex) + ") call");
                    return;
                }

            m_boxLowerStateCollectionIndex[transitionCollectionIndex] = lowerStateCollectionIndex;
            m_boxUpperStateCollectionIndex[transitionCollectionIndex] = upperStateCollectionIndex;
        }


    public:
        /** returns collection index of transition in databox
         *
         * @param lowerStateCollectionIndex collection index of lower state in a atomic state dataBox
         * @param upperStateCollectionIndex collection index of upper state in a atomic state dataBox
         * @param startIndexBlock start collection of search
         * @param numberOfTransitionsInBlock number of transitions to search
         *
         * @attention this search is slow, performant access should use collectionIndices directly
         *
         * @return returns numberTransitionsTotal if transition not found
         *
         * @todo : replace linear search with binary, Brian Marre, 2022
         */
        HDINLINE uint32_t findTransitionCollectionIndex(
            Idx const lowerStateCollectionIndex,
            Idx const upperStateCollectionIndex,
            uint32_t const numberOfTransitionsInBlock,
            uint32_t const startIndexBlock = 0u) const
        {
            // search in corresponding block in transitions box
            for(uint32_t i = 0u; i < numberOfTransitionsInBlock; i++)
            {
                if((m_boxLowerStateCollectionIndex(i + startIndexBlock) == lowerStateCollectionIndex)
                   && (m_boxUpperStateCollectionIndex(i + startIndexBlock) == upperStateCollectionIndex))
                    return i + startIndexBlock;
            }

            return m_numberTransitions;
        }

        /** returns upper states configNumber of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks
         */
        HDINLINE Idx upperStateCollectionIndex(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= m_numberTransitions)
                {
                    printf(
                        "atomicPhysics ERROR: out of range getUpperConfigNumberTransition(%u) call\n",
                        collectionIndex);
                    return static_cast<Idx>(0u);
                }

            return m_boxUpperStateCollectionIndex(collectionIndex);
        }

        /** returns lower states configNumber of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks
         */
        HDINLINE Idx lowerStateCollectionIndex(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= m_numberTransitions)
                {
                    printf(
                        "atomicPhysics ERROR: out of range getLowerConfigNumberTransition(%u) call\n",
                        collectionIndex);
                    return static_cast<Idx>(0u);
                }

            return m_boxLowerStateCollectionIndex(collectionIndex);
        }

        HDINLINE uint32_t getNumberOfTransitionsTotal() const
        {
            return m_numberTransitions;
        }
    };

    /** complementing buffer class
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_CollectionIndex dataType sed for collection index storage, typically uint32_t
     */
    template<typename T_Number, typename T_Value, typename T_CollectionIndex>
    class TransitionDataBuffer : public DataBuffer<T_Number, T_Value>
    {
    public:
        using Idx = T_CollectionIndex;
        using BufferCollectionIndex = pmacc::HostDeviceBuffer<T_CollectionIndex, 1u>;

    protected:
        std::unique_ptr<BufferCollectionIndex> bufferLowerStateCollectionIndex;
        std::unique_ptr<BufferCollectionIndex> bufferUpperStateCollectionIndex;

        uint32_t m_numberTransitions;

    public:
        /** buffer corresponding to the above dataBox object
         *
         * @param numberAtomicStates number of atomic states, and number of buffer entries
         */
        HINLINE TransitionDataBuffer(uint32_t numberTransitions)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutTransitions = pmacc::GridLayout<1>(numberTransitions, guardSize).sizeWithoutGuardND();

            bufferLowerStateCollectionIndex.reset(new BufferCollectionIndex(layoutTransitions, false));
            bufferUpperStateCollectionIndex.reset(new BufferCollectionIndex(layoutTransitions, false));

            m_numberTransitions = numberTransitions;
        }

        HINLINE TransitionDataBox<T_Number, T_Value, T_CollectionIndex> getHostDataBox()
        {
            return TransitionDataBox<T_Number, T_Value, T_CollectionIndex>(
                bufferLowerStateCollectionIndex->getHostBuffer().getDataBox(),
                bufferUpperStateCollectionIndex->getHostBuffer().getDataBox(),
                m_numberTransitions);
        }

        HINLINE TransitionDataBox<T_Number, T_Value, T_CollectionIndex> getDeviceDataBox()
        {
            return TransitionDataBox<T_Number, T_Value, T_CollectionIndex>(
                bufferLowerStateCollectionIndex->getDeviceBuffer().getDataBox(),
                bufferUpperStateCollectionIndex->getDeviceBuffer().getDataBox(),
                m_numberTransitions);
        }

        HINLINE void hostToDevice_BaseClass()
        {
            bufferLowerStateCollectionIndex->hostToDevice();
            bufferUpperStateCollectionIndex->hostToDevice();
        }

        HINLINE void deviceToHost_BaseClass()
        {
            bufferLowerStateCollectionIndex->deviceToHost();
            bufferUpperStateCollectionIndex->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics::atomicData
