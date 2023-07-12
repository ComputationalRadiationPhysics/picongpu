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

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBuffer.hpp"

#include <cstdint>
#include <memory>

namespace picongpu::particles::atomicPhysics2::atomicData
{
    /** Data box storing for each atomic state the total number of possible transitions, excluding field ionization
     *
     * The chooseTransition kernel needs to know how many transition there are
     *  for every modelled atomic state to be able to choose with equal weight.
     *
     * This data is available via the AtomicStatesNumberOfTransitionsDataBoxes,
     *  but keeping all boxes in memory requires too much memory.
     * Since only the total number is actually required we pre-compute it and load only it.
     *
     * for use on device.
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     */
    template<typename T_Number, typename T_Value>
    class TransitionSelectionDataBox : public DataBox<T_Number, T_Value>
    {
    public:
        using S_DataBox = DataBox<T_Number, T_Value>;

    private:
        //! total number of physical transitions
        typename S_DataBox::BoxNumber m_boxNumberPhysicalTransitionsTotal;

    public:
        /** constructor
         *
         * @attention atomic state data must be sorted block-wise by charge state
         *  and secondary ascending by configNumber.
         * @attention the completely ionized state must be left out.
         *
         * @param boxNumberPhysicalTransitionsTotal number of physical transitions from the atomic state
         */
        TransitionSelectionDataBox(typename S_DataBox::BoxNumber boxNumberPhysicalTransitionsTotal)
            : m_boxNumberPhysicalTransitionsTotal(boxNumberPhysicalTransitionsTotal)
        {
        }

        //! @attention no range check!
        HINLINE void store(uint32_t collectionIndex, typename S_DataBox::TypeNumber numberPhysicalTransitionsTotal)
        {
            m_boxNumberPhysicalTransitionsTotal[collectionIndex] = numberPhysicalTransitionsTotal;
        }

        /** get total number of physical transitions for a atomic state
         *
         * @param collectionIndex atomic state collection index
         *
         * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
         *
         * @attention no range check
         */
        HDINLINE typename S_DataBox::TypeNumber numberTransitions(uint32_t const collectionIndex) const
        {
            return m_boxNumberPhysicalTransitionsTotal[collectionIndex];
        }
    };

    /** complementing buffer class
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     */
    template<typename T_Number, typename T_Value>
    class TransitionSelectionDataBuffer : public DataBuffer<T_Number, T_Value>
    {
    public:
        using S_DataBuffer = DataBuffer<T_Number, T_Value>;

    private:
        //! total number of physical transitions
        std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferNumberPhysicalTransitionsTotal;

    public:
        /** buffer corresponding to the above dataBox object
         *
         * @param numberAtomicStates number of atomic states, and number of buffer entries
         */
        HINLINE TransitionSelectionDataBuffer(uint32_t numberAtomicStates)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutAtomicStates
                = pmacc::GridLayout<1>(numberAtomicStates, guardSize).getDataSpaceWithoutGuarding();

            bufferNumberPhysicalTransitionsTotal.reset(new
                                                       typename S_DataBuffer::BufferNumber(layoutAtomicStates, false));
        }

        HINLINE TransitionSelectionDataBox<T_Number, T_Value> getHostDataBox()
        {
            return TransitionSelectionDataBox<T_Number, T_Value>(
                bufferNumberPhysicalTransitionsTotal->getHostBuffer().getDataBox());
        }

        HINLINE TransitionSelectionDataBox<T_Number, T_Value> getDeviceDataBox()
        {
            return TransitionSelectionDataBox<T_Number, T_Value>(
                bufferNumberPhysicalTransitionsTotal->getDeviceBuffer().getDataBox());
        }

        HINLINE void hostToDevice()
        {
            bufferNumberPhysicalTransitionsTotal->hostToDevice();
        }

        HINLINE void deviceToHost()
        {
            bufferNumberPhysicalTransitionsTotal->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics2::atomicData
