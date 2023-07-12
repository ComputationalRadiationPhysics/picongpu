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

#include "picongpu/simulation_defines.hpp" // need: picongpu/param/atomicPhysics2_Debug.param

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics2/atomicData/TransitionData.hpp"
#include "picongpu/particles/atomicPhysics2/enums/ProcessClassGroup.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionOrdering.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>

/** @file implements the storage of bound-bound transitions property data
 */

namespace picongpu::particles::atomicPhysics2::atomicData
{
    /** data box storing bound-bound transition property data
     *
     * for use on device.
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_CollectionIndex dataType used for atomic state collectionIndex,
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
        picongpu::particles::atomicPhysics2::enums::TransitionOrdering T_TransitionOrdering>
    class BoundBoundTransitionDataBox : public TransitionDataBox<T_Number, T_Value, T_CollectionIndex>
    {
    public:
        using S_TransitionDataBox = TransitionDataBox<T_Number, T_Value, T_CollectionIndex>;
        using S_BoundBoundTransitionTuple
            = BoundBoundTransitionTuple<typename S_TransitionDataBox::TypeValue, T_ConfigNumberDataType>;

        static constexpr auto processClassGroup = particles::atomicPhysics2::enums::ProcessClassGroup::boundBoundBased;
        static constexpr auto transitionOrdering = T_TransitionOrdering;

    private:
        //! unitless
        typename S_TransitionDataBox::S_DataBox::BoxValue m_boxCollisionalOscillatorStrength;
        //! unitless
        typename S_TransitionDataBox::S_DataBox::BoxValue m_boxAbsorptionOscillatorStrength;
        //! gaunt tunneling fit parameter 1, unitless
        typename S_TransitionDataBox::S_DataBox::BoxValue m_boxCxin1;
        //! gaunt tunneling fit parameter 2, unitless
        typename S_TransitionDataBox::S_DataBox::BoxValue m_boxCxin2;
        //! gaunt tunneling fit parameter 3, unitless
        typename S_TransitionDataBox::S_DataBox::BoxValue m_boxCxin3;
        //! gaunt tunneling fit parameter 4, unitless
        typename S_TransitionDataBox::S_DataBox::BoxValue m_boxCxin4;
        //! gaunt tunneling fit parameter 5, unitless
        typename S_TransitionDataBox::S_DataBox::BoxValue m_boxCxin5;

    public:
        /** constructor
         *
         * @attention transition data must be sorted block-wise ascending by lower/upper
         *  atomic state and secondary ascending by upper/lower atomic state.
         *
         * @param boxCollisionalOscillatorStrength
         * @param boxAbsorptionOscillatorStrength
         * @param boxCxin1 gaunt tunnelling fit parameter 1
         * @param boxCxin2 gaunt tunnelling fit parameter 2
         * @param boxCxin3 gaunt tunnelling fit parameter 3
         * @param boxCxin4 gaunt tunnelling fit parameter 4
         * @param boxCxin5 gaunt tunnelling fit parameter 5
         * @param boxLowerStateCollectionIndex collection Index of the lower
         *      (lower excitation energy) state of the transition in an atomic state dataBox
         * @param boxUpperStateCollectionIndex collection Index of the upper
         *      (higher excitation energy) state of the transition in an atomic state dataBox
         * @param numberTransitions number of atomic bound-bound transitions stored
         */
        BoundBoundTransitionDataBox(
            typename S_TransitionDataBox::S_DataBox::BoxValue boxCollisionalOscillatorStrength,
            typename S_TransitionDataBox::S_DataBox::BoxValue boxAbsorptionOscillatorStrength,
            typename S_TransitionDataBox::S_DataBox::BoxValue boxCxin1,
            typename S_TransitionDataBox::S_DataBox::BoxValue boxCxin2,
            typename S_TransitionDataBox::S_DataBox::BoxValue boxCxin3,
            typename S_TransitionDataBox::S_DataBox::BoxValue boxCxin4,
            typename S_TransitionDataBox::S_DataBox::BoxValue boxCxin5,
            typename S_TransitionDataBox::BoxCollectionIndex boxLowerStateCollectionIndex,
            typename S_TransitionDataBox::BoxCollectionIndex boxUpperStateCollectionIndex,
            uint32_t numberTransitions)
            : TransitionDataBox<T_Number, T_Value, T_CollectionIndex>(
                boxLowerStateCollectionIndex,
                boxUpperStateCollectionIndex,
                numberTransitions)
            , m_boxCollisionalOscillatorStrength(boxCollisionalOscillatorStrength)
            , m_boxAbsorptionOscillatorStrength(boxAbsorptionOscillatorStrength)
            , m_boxCxin1(boxCxin1)
            , m_boxCxin2(boxCxin2)
            , m_boxCxin3(boxCxin3)
            , m_boxCxin4(boxCxin4)
            , m_boxCxin5(boxCxin5)
        {
        }

        /** store transition in data box
         *
         * @attention do not forget to call syncToDevice() on the
         *  corresponding buffer, or the state is only added on the host side.
         * @attention needs to fulfill all ordering and content assumptions of constructor!
         * @attention no range checks outside debug compile, invalid memory access if collectionIndex >=
         *  numberTransitions
         *
         * @tparam T_StateHostBox type of atomic state data box
         *
         * @param collectionIndex index of data box entry to rewrite
         * @param tuple tuple containing data of transition
         * @param stateHostBox host data box of all known atomic states,
         *      used to convert configNumber to collectionIndex, for storeTransition() call
         */
        template<typename T_StateHostBox>
        HINLINE void store(
            uint32_t const collectionIndex,
            S_BoundBoundTransitionTuple& tuple,
            T_StateHostBox const stateHostBox)
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_LOAD)
                if(collectionIndex >= this->m_numberTransitions)
                {
                    throw std::runtime_error("atomicPhysics ERROR: out of range store() bound-bound");
                    return;
                }

            m_boxCollisionalOscillatorStrength[collectionIndex] = std::get<0>(tuple);
            m_boxAbsorptionOscillatorStrength[collectionIndex] = std::get<1>(tuple);
            m_boxCxin1[collectionIndex] = std::get<2>(tuple);
            m_boxCxin2[collectionIndex] = std::get<3>(tuple);
            m_boxCxin3[collectionIndex] = std::get<4>(tuple);
            m_boxCxin4[collectionIndex] = std::get<5>(tuple);
            m_boxCxin5[collectionIndex] = std::get<6>(tuple);

            // find collection Indices
            uint32_t lowerStateCollIdx
                = stateHostBox.findStateCollectionIndex(std::get<7>(tuple), stateHostBox.numberAtomicStatesTotal());
            uint32_t upperStateCollIdx
                = stateHostBox.findStateCollectionIndex(std::get<8>(tuple), stateHostBox.numberAtomicStatesTotal());
            this->storeTransition(collectionIndex, lowerStateCollIdx, upperStateCollIdx);
        }

        /** returns collisional oscillator strength of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks outside debug compile, invalid memory access if collectionIndex >=
         * numberTransitions
         */
        HDINLINE typename S_TransitionDataBox::S_DataBox::TypeValue collisionalOscillatorStrength(
            uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= this->m_numberTransitions)
                {
                    printf("atomicPhysics ERROR: out of range call collisionalOscillatorStrength\n");
                    return static_cast<typename S_TransitionDataBox::S_DataBox::TypeValue>(0._X);
                }

            return m_boxCollisionalOscillatorStrength(collectionIndex);
        }

        /** returns absorption oscillator strength of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks outside debug compile, invalid memory access if collectionIndex >=
         * numberTransitions
         */
        HDINLINE typename S_TransitionDataBox::S_DataBox::TypeValue absorptionOscillatorStrength(
            uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= this->m_numberTransitions)
                {
                    printf("atomicPhysics ERROR: out of range call absorptionOscillatorStrength\n");
                    return static_cast<typename S_TransitionDataBox::S_DataBox::TypeValue>(0._X);
                }

            return m_boxAbsorptionOscillatorStrength(collectionIndex);
        }

        /// @todo find way to replace Cxin getters with single template function

        /** returns gaunt tunneling fit parameter 1 of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks outside debug compile, invalid memory access if collectionIndex >=
         * numberTransitions
         */
        HDINLINE typename S_TransitionDataBox::S_DataBox::TypeValue cxin1(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= this->m_numberTransitions)
                {
                    printf("atomicPhysics ERROR: out of range bound-bound cxin1() call\n");
                    return static_cast<typename S_TransitionDataBox::S_DataBox::TypeValue>(0._X);
                }

            return m_boxCxin1(collectionIndex);
        }

        /** returns gaunt tunneling fit parameter 2 of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks outside debug compile, invalid memory access if collectionIndex >=
         * numberTransitions
         */
        HDINLINE typename S_TransitionDataBox::S_DataBox::TypeValue cxin2(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= this->m_numberTransitions)
                {
                    printf("atomicPhysics ERROR: out of range bound-bound cxin2() call\n");
                    return static_cast<typename S_TransitionDataBox::S_DataBox::TypeValue>(0._X);
                }

            return m_boxCxin2(collectionIndex);
        }

        /** returns gaunt tunneling fit parameter 3 of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks outside debug compile, invalid memory access if collectionIndex >=
         * numberTransitions
         */
        HDINLINE typename S_TransitionDataBox::S_DataBox::TypeValue cxin3(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= this->m_numberTransitions)
                {
                    printf("atomicPhysics ERROR: out of range bound-bound cxin3() call\n");
                    return static_cast<typename S_TransitionDataBox::S_DataBox::TypeValue>(0._X);
                }

            return m_boxCxin3(collectionIndex);
        }

        /** returns gaunt tunneling fit parameter 4 of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks outside debug compile, invalid memory access if collectionIndex >=
         * numberTransitions
         */
        HDINLINE typename S_TransitionDataBox::S_DataBox::TypeValue cxin4(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= this->m_numberTransitions)
                {
                    printf("atomicPhysics ERROR: out of range bound-bound cxin4() call\n");
                    return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                }

            return m_boxCxin4(collectionIndex);
        }

        /** returns gaunt tunneling fit parameter 5 of the transition
         *
         * @param collectionIndex ... collection index of transition
         *
         * @attention no range checks outside debug compile, invalid memory access if collectionIndex >=
         * numberTransitions
         */
        HDINLINE typename S_TransitionDataBox::S_DataBox::TypeValue cxin5(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= this->m_numberTransitions)
                {
                    printf("atomicPhysics ERROR: out of range bound-bound cxin5() call\n");
                    return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                }

            return m_boxCxin5(collectionIndex);
        }
    };

    /** complementing buffer class
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_CollectionIndex used for index storage, typically uint32_t
     * @tparam T_ConfigNumberDataType dataType used for configNumber storage, typically uint64_t
     * @tparam T_TransitionOrdering ordering used for data
     */
    template<
        typename T_Number,
        typename T_Value,
        typename T_CollectionIndex,
        typename T_ConfigNumberDataType,
        picongpu::particles::atomicPhysics2::enums::TransitionOrdering T_TransitionOrdering>
    class BoundBoundTransitionDataBuffer : public TransitionDataBuffer<T_Number, T_Value, T_CollectionIndex>
    {
    public:
        using S_TransitionDataBuffer = TransitionDataBuffer<T_Number, T_Value, T_CollectionIndex>;
        using DataBoxType = BoundBoundTransitionDataBox<
            T_Number,
            T_Value,
            T_CollectionIndex,
            T_ConfigNumberDataType,
            T_TransitionOrdering>;

        static constexpr auto processClassGroup = particles::atomicPhysics2::enums::ProcessClassGroup::boundBoundBased;
        static constexpr auto transitionOrdering = T_TransitionOrdering;

    private:
        std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCollisionalOscillatorStrength;
        std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferAbsorptionOscillatorStrength;
        std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin1;
        std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin2;
        std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin3;
        std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin4;
        std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin5;

    public:
        /** buffer corresponding to the above dataBox object
         *
         * @param numberAtomicStates number of atomic states, and number of buffer entries
         */
        HINLINE BoundBoundTransitionDataBuffer(uint32_t numberBoundBoundTransitions)
            : TransitionDataBuffer<T_Number, T_Value, T_CollectionIndex>(numberBoundBoundTransitions)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutBoundBoundTransitions
                = pmacc::GridLayout<1>(numberBoundBoundTransitions, guardSize).getDataSpaceWithoutGuarding();

            bufferCollisionalOscillatorStrength.reset(
                new typename S_TransitionDataBuffer::BufferValue(layoutBoundBoundTransitions, false));
            bufferAbsorptionOscillatorStrength.reset(
                new typename S_TransitionDataBuffer::BufferValue(layoutBoundBoundTransitions, false));
            bufferCxin1.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundBoundTransitions, false));
            bufferCxin2.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundBoundTransitions, false));
            bufferCxin3.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundBoundTransitions, false));
            bufferCxin4.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundBoundTransitions, false));
            bufferCxin5.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundBoundTransitions, false));
        }

        HINLINE DataBoxType getHostDataBox()
        {
            return DataBoxType(
                bufferCollisionalOscillatorStrength->getHostBuffer().getDataBox(),
                bufferAbsorptionOscillatorStrength->getHostBuffer().getDataBox(),
                bufferCxin1->getHostBuffer().getDataBox(),
                bufferCxin2->getHostBuffer().getDataBox(),
                bufferCxin3->getHostBuffer().getDataBox(),
                bufferCxin4->getHostBuffer().getDataBox(),
                bufferCxin5->getHostBuffer().getDataBox(),
                this->bufferLowerStateCollectionIndex->getHostBuffer().getDataBox(),
                this->bufferUpperStateCollectionIndex->getHostBuffer().getDataBox(),
                this->m_numberTransitions);
        }

        HINLINE DataBoxType getDeviceDataBox()
        {
            return DataBoxType(
                bufferCollisionalOscillatorStrength->getDeviceBuffer().getDataBox(),
                bufferAbsorptionOscillatorStrength->getDeviceBuffer().getDataBox(),
                bufferCxin1->getDeviceBuffer().getDataBox(),
                bufferCxin2->getDeviceBuffer().getDataBox(),
                bufferCxin3->getDeviceBuffer().getDataBox(),
                bufferCxin4->getDeviceBuffer().getDataBox(),
                bufferCxin5->getDeviceBuffer().getDataBox(),
                this->bufferLowerStateCollectionIndex->getDeviceBuffer().getDataBox(),
                this->bufferUpperStateCollectionIndex->getDeviceBuffer().getDataBox(),
                this->m_numberTransitions);
        }

        HINLINE void hostToDevice()
        {
            bufferCollisionalOscillatorStrength->hostToDevice();
            bufferAbsorptionOscillatorStrength->hostToDevice();
            bufferCxin1->hostToDevice();
            bufferCxin2->hostToDevice();
            bufferCxin3->hostToDevice();
            bufferCxin4->hostToDevice();
            bufferCxin5->hostToDevice();
            this->hostToDevice_BaseClass();
        }

        HINLINE void deviceToHost()
        {
            bufferCollisionalOscillatorStrength->deviceToHost();
            bufferAbsorptionOscillatorStrength->deviceToHost();
            bufferCxin1->deviceToHost();
            bufferCxin2->deviceToHost();
            bufferCxin3->deviceToHost();
            bufferCxin4->deviceToHost();
            bufferCxin5->deviceToHost();
            this->deviceToHost_BaseClass();
        }
    };
} // namespace picongpu::particles::atomicPhysics2::atomicData
