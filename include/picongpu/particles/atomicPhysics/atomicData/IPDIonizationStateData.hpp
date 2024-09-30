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

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"

#include <cstdint>

/** @file implements storage of pressure ionization data
 *
 * stores for each atomic state the collection index of its pressure ionization state
 */

namespace picongpu::particles::atomicPhysics::atomicData
{
    /** data box storing pressure ionization state for each atomic state
     *
     * stores collectionIndex of pressure ionization state for each atomic state
     *
     * @tparam T_CollectionIndexType dataType used for atomicState collectionIndex
     */
    template<typename T_CollectionIndexType>
    struct IPDIonizationStateDataBox
    {
        using CollectionIdx = T_CollectionIndexType;
        using BoxCollectionIndex = pmacc::DataBox<pmacc::PitchedBox<T_CollectionIndexType, 1u>>;

    private:
        //! collectionIndex of pressure ionization state for each atomic state
        BoxCollectionIndex m_boxCollectionIndex;
        uint32_t m_numberAtomicStates;

    public:
        /** constructor
         *
         * @param boxCollectionIndex dataBox of pressure ionization state collection index
         * @param numberAtomicStates number of atomic states
         */
        IPDIonizationStateDataBox(BoxCollectionIndex boxCollectionIndex, uint32_t numberAtomicStates)
            : m_boxCollectionIndex(boxCollectionIndex)
            , m_numberAtomicStates(numberAtomicStates)
        {
        }

        /** store pressure ionization state collectionIndex for the given atomic state
         *
         * @attention do not forget to call syncToDevice() on the
         *  corresponding buffer, or the state is only stored on the host side.
         * @attention no range check invalid memory access if collectionIndex >= numberAtomicStates
         *
         * @param state collectionIndex of an atomic state
         * @param IPDIonizationState collectionIndex of it's pressure IPDIonizationState
         */
        HINLINE void store(CollectionIdx const state, CollectionIdx const ipdIonizationState)
        {
            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(state >= m_numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of bounds in store() call on IPDIonizationData\n");
                }
            m_boxCollectionIndex[state] = ipdIonizationState;
        }

        /** get collectionIndex of pressure ionization state for state
         *
         * @param state collectionIndex of an atomic state
         */
        HDINLINE CollectionIdx ipdIonizationState(CollectionIdx const state) const
        {
            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(state >= m_numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of bounds in presssureIonizationState() call\n");
                    return static_cast<CollectionIdx>(0u);
                }
            return m_boxCollectionIndex[state];
        }
    };

    /** complementing buffer class
     *
     * @tparam T_CollectionIndexType dataType used for atomicState collectionIndex
     */
    template<typename T_CollectionIndexType>
    struct IPDIonizationStateDataBuffer
    {
        using CollectionIdx = T_CollectionIndexType;
        using BufferCollectionIndex = pmacc::HostDeviceBuffer<CollectionIdx, 1u>;

    private:
        std::unique_ptr<BufferCollectionIndex> bufferCollectionIndex;
        uint32_t m_numberAtomicStates;

    public:
        HINLINE IPDIonizationStateDataBuffer(uint32_t numberAtomicStates) : m_numberAtomicStates(numberAtomicStates)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutAtomicStates = pmacc::GridLayout<1>(numberAtomicStates, guardSize).sizeWithoutGuardND();
            bufferCollectionIndex.reset(new BufferCollectionIndex(layoutAtomicStates, false));
        }

        HINLINE IPDIonizationStateDataBox<CollectionIdx> getHostDataBox()
        {
            return IPDIonizationStateDataBox<CollectionIdx>(
                bufferCollectionIndex->getHostBuffer().getDataBox(),
                m_numberAtomicStates);
        }

        HINLINE IPDIonizationStateDataBox<CollectionIdx> getDeviceDataBox()
        {
            return IPDIonizationStateDataBox<CollectionIdx>(
                bufferCollectionIndex->getDeviceBuffer().getDataBox(),
                m_numberAtomicStates);
        }

        HINLINE void hostToDevice()
        {
            bufferCollectionIndex->hostToDevice();
        }

        HINLINE void deviceToHost()
        {
            bufferCollectionIndex->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics::atomicData
