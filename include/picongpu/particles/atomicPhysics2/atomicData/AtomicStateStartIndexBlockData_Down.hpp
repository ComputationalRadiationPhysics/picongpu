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

#include "picongpu/particles/atomicPhysics2/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBuffer.hpp"
#include "picongpu/particles/atomicPhysics2/enums/ProcessClassGroup.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>

/** @file implements base class of atomic state start index block data with up- and downward transitions
 *
 * e.g. for autonomous transitions
 */

namespace picongpu::particles::atomicPhysics2::atomicData
{
    /** data box storing for each atomic state the startIndexBlock for downward-only transitions
     *
     * for use on device.
     *
     * @tparam T_CollectionIndex dataType of collectionIndex, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_ProcessClassGroup processClassGroup current data corresponds to
     */
    template<
        typename T_CollectionIndex,
        typename T_Value,
        particles::atomicPhysics2::enums::ProcessClassGroup T_ProcessClassGroup>
    class AtomicStateStartIndexBlockDataBox_Down : public DataBox<T_CollectionIndex, T_Value>
    {
    public:
        using S_DataBox = DataBox<T_CollectionIndex, T_Value>;
        static constexpr auto processClassGroup = T_ProcessClassGroup;

    private:
        /** start collection index of the block of autonomous transitions
         * from the atomic state in the collection of autonomous transitions
         */
        typename S_DataBox::BoxNumber m_boxStartIndexBlockTransitions;

        /// @todo transitions from configNumber 0u?, Brian Marre, 2022

    public:
        /** constructor
         *
         * @attention atomic state data must be sorted block-wise by charge state
         *  and secondary ascending by configNumber.
         *
         * @param boxNumberTransitions number of autonomous transitions from the atomic state
         * @param startIndexBlockAtomicStates start collection index of block of
         *  autonomous transitions in autonomous transition collection
         */
        AtomicStateStartIndexBlockDataBox_Down(typename S_DataBox::BoxNumber boxStartIndexBlockTransitions)
            : m_boxStartIndexBlockTransitions(boxStartIndexBlockTransitions)
        {
        }

        //! @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
        ALPAKA_FN_HOST void storeDown(uint32_t const collectionIndex, typename S_DataBox::TypeNumber startIndexDown)
        {
            m_boxStartIndexBlockTransitions[collectionIndex] = startIndexDown;
        }

        /** get start index of block of autonomous transitions from atomic state
         *
         * @param collectionIndex atomic state collection index
         *
         * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
         * @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
         */
        HDINLINE typename S_DataBox::TypeNumber startIndexBlockTransitionsDown(uint32_t const collectionIndex) const
        {
            return m_boxStartIndexBlockTransitions(collectionIndex);
        }
    };

    /** complementing buffer class
     *
     * @tparam T_CollectionIndex dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_ProcessClassGroup processClassGroup current data corresponds to
     */
    template<
        typename T_CollectionIndex,
        typename T_Value,
        particles::atomicPhysics2::enums::ProcessClassGroup T_ProcessClassGroup>
    class AtomicStateStartIndexBlockDataBuffer_Down : public DataBuffer<T_CollectionIndex, T_Value>
    {
    public:
        using DataBoxType = AtomicStateStartIndexBlockDataBox_Down<T_CollectionIndex, T_Value, T_ProcessClassGroup>;
        using S_DataBuffer = DataBuffer<T_CollectionIndex, T_Value>;

    private:
        std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferStartIndexBlockTransitionsDown;

    public:
        HINLINE AtomicStateStartIndexBlockDataBuffer_Down(uint32_t numberAtomicStates)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutAtomicStates
                = pmacc::GridLayout<1>(numberAtomicStates, guardSize).getDataSpaceWithoutGuarding();

            bufferStartIndexBlockTransitionsDown.reset(new
                                                       typename S_DataBuffer::BufferNumber(layoutAtomicStates, false));
        }

        HINLINE DataBoxType getHostDataBox()
        {
            return DataBoxType(bufferStartIndexBlockTransitionsDown->getHostBuffer().getDataBox());
        }

        HINLINE DataBoxType getDeviceDataBox()
        {
            return DataBoxType(bufferStartIndexBlockTransitionsDown->getDeviceBuffer().getDataBox());
        }

        HINLINE void hostToDevice()
        {
            bufferStartIndexBlockTransitionsDown->hostToDevice();
        }

        HINLINE void deviceToHost()
        {
            bufferStartIndexBlockTransitionsDown->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics2::atomicData
