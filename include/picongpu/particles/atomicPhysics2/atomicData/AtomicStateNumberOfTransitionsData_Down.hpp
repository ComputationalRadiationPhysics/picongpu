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

/** @file implements storage of numberInBlock data for each atomic state with downward transitions
 *
 * e.g. for autonomous transitions
 */

namespace picongpu::particles::atomicPhysics2::atomicData
{
    /** data box storing atomic state numberInBlock for downward-only transitions
     *
     * for use on device.
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_ProcessClassGroup processClassGroup current data corresponds to
     */
    template<
        typename T_Number,
        typename T_Value,
        particles::atomicPhysics2::enums::ProcessClassGroup T_ProcessClassGroup>
    class AtomicStateNumberOfTransitionsDataBox_Down : public DataBox<T_Number, T_Value>
    {
    public:
        using S_DataBox = DataBox<T_Number, T_Value>;
        static constexpr auto processClassGroup = T_ProcessClassGroup;

    private:
        /** start collection index of the block of autonomous transitions
         * from the atomic state in the collection of autonomous transitions
         */
        typename S_DataBox::BoxNumber m_boxNumberOfTransitions;

        /// @todo transitions from configNumber 0u?

    public:
        /** constructor
         *
         * @attention atomic state data must be sorted block-wise by charge state
         *  and secondary ascending by configNumber.
         *
         * @param boxNumberTransitions number of transitions from the atomic state
         */
        AtomicStateNumberOfTransitionsDataBox_Down(typename S_DataBox::BoxNumber boxNumberOfTransitions)
            : m_boxNumberOfTransitions(boxNumberOfTransitions)
        {
        }

        //! @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
        ALPAKA_FN_HOST void storeDown(uint32_t const collectionIndex, typename S_DataBox::TypeNumber const numberDown)
        {
            m_boxNumberOfTransitions[collectionIndex] = numberDown;
        }

        /** get number of transitions from atomic state
         *
         * @param collectionIndex atomic state collection index
         *
         * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
         * @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
         */
        HDINLINE typename S_DataBox::TypeNumber numberOfTransitionsDown(uint32_t const collectionIndex) const
        {
            return m_boxNumberOfTransitions(collectionIndex);
        }
    };

    /** complementing buffer class
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_ProcessClassGroup processClassGroup current data corresponds to
     */
    template<
        typename T_Number,
        typename T_Value,
        particles::atomicPhysics2::enums::ProcessClassGroup T_ProcessClassGroup>
    class AtomicStateNumberOfTransitionsDataBuffer_Down : public DataBuffer<T_Number, T_Value>
    {
    public:
        using S_DataBuffer = DataBuffer<T_Number, T_Value>;
        using DataBoxType = AtomicStateNumberOfTransitionsDataBox_Down<T_Number, T_Value, T_ProcessClassGroup>;

    private:
        std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferNumberOfTransitionsBlockTransitionsDown;

    public:
        HINLINE AtomicStateNumberOfTransitionsDataBuffer_Down(uint32_t numberAtomicStates)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutAtomicStates
                = pmacc::GridLayout<1>(numberAtomicStates, guardSize).getDataSpaceWithoutGuarding();

            bufferNumberOfTransitionsBlockTransitionsDown.reset(
                new typename S_DataBuffer::BufferNumber(layoutAtomicStates, false));
        }

        HINLINE DataBoxType getHostDataBox()
        {
            return DataBoxType(bufferNumberOfTransitionsBlockTransitionsDown->getHostBuffer().getDataBox());
        }

        HINLINE DataBoxType getDeviceDataBox()
        {
            return DataBoxType(bufferNumberOfTransitionsBlockTransitionsDown->getDeviceBuffer().getDataBox());
        }

        HINLINE void hostToDevice()
        {
            bufferNumberOfTransitionsBlockTransitionsDown->hostToDevice();
        }

        HINLINE void deviceToHost()
        {
            bufferNumberOfTransitionsBlockTransitionsDown->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics2::atomicData
