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
// need: picongpu/param/atomicPhysics_Debug.param and */unit.param

#include "picongpu/particles/atomicPhysics/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics/atomicData/TransitionData.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClassGroup.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>

/** @file implements the storage of autonomous transitions property data
 */

namespace picongpu::particles::atomicPhysics::atomicData
{
    /** data box storing bound-free transition property data
     *
     * for use on device.
     *
     * @tparam T_DataBoxType dataBox type used for storage
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_CollectionIndex dataType used for collection index,
     *      typically uint32_t
     * @tparam T_TransitionIndexDataType dataType used for transition index,
     *      typically uint32_t
     * @tparam T_ConfigNumberDataType dataType used for configNumber storage,
     *      typically uint64_t
     * @tparam T_TransitionOrdering ordering used for data
     *
     * @attention ConfigNumber specifies the number of a state as defined by the configNumber
     *      class, while index always refers to a collection index.
     *      The configNumber of a given state is always the same, its collection index depends
     *      on input file,it should therefore only be used internally!
     */
    template<
        typename T_Number,
        typename T_Value,
        typename T_CollectionIndex,
        typename T_ConfigNumberDataType,
        picongpu::particles::atomicPhysics::enums::TransitionOrdering T_TransitionOrdering>
    class AutonomousTransitionDataBox : public TransitionDataBox<T_Number, T_Value, T_CollectionIndex>
    {
    public:
        using S_TransitionDataBox = TransitionDataBox<T_Number, T_Value, T_CollectionIndex>;
        using S_AutonomousTransitionTuple = AutonomousTransitionTuple<T_ConfigNumberDataType>;

        static constexpr auto processClassGroup = particles::atomicPhysics::enums::ProcessClassGroup::autonomousBased;
        static constexpr auto transitionOrdering = T_TransitionOrdering;

    private:
        typename S_TransitionDataBox::S_DataBox::BoxValue m_boxTransitionRate; // unit: 1/UNIT_TIME

    public:
        /** constructor
         *
         * @attention transition data must be sorted block-wise ascending by lower/upper
         *  atomic state and secondary ascending by upper/lower atomic state.
         *
         * @param boxTransitionRate rate over deexcitation [1/s]
         * @param boxLowerStateCollectionIndex collection index of the lower
         *    (lower energy) state of the transition in an atomic state dataBox
         * @param boxUpperStateCollectionIndex collection index of the upper
         *    (higher energy) state of the transition in an atomic state dataBox
         * @param numberTransitions number of atomic autonomous transitions stored
         */
        AutonomousTransitionDataBox(
            typename S_TransitionDataBox::S_DataBox::BoxValue boxTransitionRate,
            typename S_TransitionDataBox::BoxCollectionIndex boxLowerStateCollectionIndex,
            typename S_TransitionDataBox::BoxCollectionIndex boxUpperStateCollectionIndex,
            uint32_t numberTransitions)
            : TransitionDataBox<T_Number, T_Value, T_CollectionIndex>(
                boxLowerStateCollectionIndex,
                boxUpperStateCollectionIndex,
                numberTransitions)
            , m_boxTransitionRate(boxTransitionRate)
        {
        }

        /** store transition in data box
         *
         * @attention do not forget to call syncToDevice() on the
         *  corresponding buffer, or the state is only added on the host side.
         * @attention needs to fulfil all ordering and content assumptions of constructor!
         * @attention no range checks outside of debug compile, invalid memory access if collectionIndex >=
         * numberTransitions
         *
         * @param collectionIndex index of data box entry to rewrite
         * @param tuple tuple containing data of transition
         */
        template<typename T_StateHostBox>
        ALPAKA_FN_HOST void store(
            uint32_t const collectionIndex,
            S_AutonomousTransitionTuple& tuple,
            T_StateHostBox const stateHostBox)
        {
            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_LOAD)
                if(collectionIndex >= S_TransitionDataBox::m_numberTransitions)
                {
                    throw std::runtime_error("atomicPhysics ERROR: out of range store() call");
                    return;
                }

            // 1/s * s/UNIT_TIME) = s/s * 1/UNIT_TIME = 1/UNIT_TIME
            m_boxTransitionRate[collectionIndex]
                = static_cast<float_X>(std::get<0>(tuple) * picongpu::UNIT_TIME); // 1/UNIT_TIME

            // find collection indices
            uint32_t lowerStateCollIdx
                = stateHostBox.findStateCollectionIndex(std::get<1>(tuple), stateHostBox.numberAtomicStatesTotal());
            uint32_t upperStateCollIdx
                = stateHostBox.findStateCollectionIndex(std::get<2>(tuple), stateHostBox.numberAtomicStatesTotal());
            this->storeTransition(collectionIndex, lowerStateCollIdx, upperStateCollIdx);
        }

        /** returns rate of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks outside of debug compile, invalid memory access if collectionIndex >=
         * numberTransitions
         */
        HDINLINE typename S_TransitionDataBox::S_DataBox::TypeValue rate(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= S_TransitionDataBox::m_numberTransitions)
                {
                    printf("atomicPhysics ERROR: out of range getTransitionRate() call\n");
                    return static_cast<typename S_TransitionDataBox::S_DataBox::TypeValue>(0._X);
                }

            return m_boxTransitionRate(collectionIndex);
        }
    };

    /** complementing buffer class
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_CollectionIndex data type used for configNumber storage, typically uint32_t
     * @tparam T_ConfigNumberDataType dataType used for configNumber storage, typically uint64_t
     * @tparam T_TransitionOrdering ordering used for data
     */
    template<
        typename T_Number,
        typename T_Value,
        typename T_CollectionIndex,
        typename T_ConfigNumberDataType,
        picongpu::particles::atomicPhysics::enums::TransitionOrdering T_TransitionOrdering>
    class AutonomousTransitionDataBuffer : public TransitionDataBuffer<T_Number, T_Value, T_CollectionIndex>
    {
    public:
        using S_TransitionDataBuffer = TransitionDataBuffer<T_Number, T_Value, T_CollectionIndex>;
        using DataBoxType = AutonomousTransitionDataBox<
            T_Number,
            T_Value,
            T_CollectionIndex,
            T_ConfigNumberDataType,
            T_TransitionOrdering>;

        static constexpr auto processClassGroup = particles::atomicPhysics::enums::ProcessClassGroup::autonomousBased;
        static constexpr auto transitionOrdering = T_TransitionOrdering;

    private:
        std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferTransitionRate;

    public:
        /** buffer corresponding to the above dataBox object
         *
         * @param numberAtomicStates number of atomic states, and number of buffer entries
         */
        HINLINE AutonomousTransitionDataBuffer(uint32_t numberAutonomousTransitions)
            : TransitionDataBuffer<T_Number, T_Value, T_CollectionIndex>(numberAutonomousTransitions)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutAutonomousTransitions
                = pmacc::GridLayout<1>(numberAutonomousTransitions, guardSize).sizeWithoutGuardND();

            bufferTransitionRate.reset(
                new typename S_TransitionDataBuffer::BufferValue(layoutAutonomousTransitions, false));
        }

        HINLINE DataBoxType getHostDataBox()
        {
            return DataBoxType(
                bufferTransitionRate->getHostBuffer().getDataBox(),
                this->bufferLowerStateCollectionIndex->getHostBuffer().getDataBox(),
                this->bufferUpperStateCollectionIndex->getHostBuffer().getDataBox(),
                this->m_numberTransitions);
        }

        HINLINE DataBoxType getDeviceDataBox()
        {
            return DataBoxType(
                bufferTransitionRate->getDeviceBuffer().getDataBox(),
                this->bufferLowerStateCollectionIndex->getDeviceBuffer().getDataBox(),
                this->bufferUpperStateCollectionIndex->getDeviceBuffer().getDataBox(),
                this->m_numberTransitions);
        }

        HINLINE void hostToDevice()
        {
            bufferTransitionRate->hostToDevice();
            this->hostToDevice_BaseClass();
        }

        HINLINE void deviceToHost()
        {
            bufferTransitionRate->deviceToHost();
            this->deviceToHost_BaseClass();
        }
    };
} // namespace picongpu::particles::atomicPhysics::atomicData
