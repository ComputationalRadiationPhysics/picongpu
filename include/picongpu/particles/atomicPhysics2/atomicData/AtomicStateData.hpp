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
#include "picongpu/particles/atomicPhysics2/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBuffer.hpp"
#include "picongpu/particles/atomicPhysics2/rateCalculation/Multiplicities.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>

//! @file implements the storage of atomic state property data


namespace picongpu::particles::atomicPhysics2::atomicData
{
    /** data box storing state property data
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_ConfigNumber dataType used for storage of configNumber of atomic states
     * @tparam T_Multiplicity dataType used for T_Multiplicity storage, typically uint64_t
     *
     * @attention ConfigNumber specifies the number of a state as defined by the configNumber
     *      class, while index always refers to a collection index.
     *      The configNumber of a given state is always the same, its collection index depends
     *      on input file,it should therefore only be used internally!
     */
    template<typename T_Number, typename T_Value, typename T_ConfigNumber, typename T_Multiplicity>
    class AtomicStateDataBox : public DataBox<T_Number, T_Value>
    {
    public:
        //! basic data type of configNumber
        using Idx = typename T_ConfigNumber::DataType;
        //! basic dataType of multiplicity
        using TypeMultiplicity = T_Multiplicity;
        using BoxMultiplicity = pmacc::DataBox<pmacc::PitchedBox<TypeMultiplicity, 1u>>;

        //! wrapper data type with conversion methods
        using ConfigNumber = T_ConfigNumber;
        using BoxConfigNumber = pmacc::DataBox<pmacc::PitchedBox<typename T_ConfigNumber::DataType, 1u>>;

        using S_AtomicStateTuple = AtomicStateTuple<T_Value, Idx>;
        using S_DataBox = DataBox<T_Number, T_Value>;

    private:
        //! configNumber of atomic state, sorted block wise by ionization state
        BoxConfigNumber m_boxConfigNumber;
        //! energy respective to ground state of ionization state[eV], sorted block wise by ionization state
        typename S_DataBox::BoxValue m_boxStateEnergy; // eV
        //! number of physical configurations associated with state, sorted block wise by ionization state
        BoxMultiplicity m_boxMultiplicity;
        //! number of atomic states
        uint32_t m_numberAtomicStates;

    public:
        /** constructor
         *
         * @attention atomic state data must be sorted block-wise ascending by
         *  charge state and secondary ascending by configNumber.
         *
         * @param boxConfigNumber dataBox of atomic state configNumber(fancy index)
         * @param boxStateEnergy dataBox of energy respective to ground state of ionization state [eV]
         * @param boxMultiplicity dataBox of number of physical states associated with state
         * @param numberAtomicStates number of atomic states
         */
        AtomicStateDataBox(
            BoxConfigNumber boxConfigNumber,
            typename S_DataBox::BoxValue boxStateEnergy,
            BoxMultiplicity boxMultiplicity,
            uint32_t numberAtomicStates)
            : m_boxConfigNumber(boxConfigNumber)
            , m_boxStateEnergy(boxStateEnergy)
            , m_boxMultiplicity(boxMultiplicity)
            , m_numberAtomicStates(numberAtomicStates)
        {
        }

        /** store atomic state in data box
         *
         * @attention do not forget to call syncToDevice() on the
         *  corresponding buffer, or the state is only added on the host side.
         * @attention needs to fulfil all ordering and content assumptions of constructor!
         * @attention no range check outside debug compile, invalid memory access if collectionIndex >=
         *  numberAtomicStates
         *
         * @param collectionIndex index of data box entry to rewrite
         * @param tuple tuple containing data of atomic state
         */
        HINLINE void store(uint32_t const collectionIndex, S_AtomicStateTuple& tuple)
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_LOAD)
                if(collectionIndex >= m_numberAtomicStates)
                {
                    throw std::runtime_error("atomicPhysics ERROR: out of bounds atomic state store call");
                    return;
                }

            auto const configNumber = std::get<0>(tuple);
            m_boxConfigNumber[collectionIndex] = configNumber;
            m_boxStateEnergy[collectionIndex] = std::get<1>(tuple);

            // calculate multiplicity and store result
            m_boxMultiplicity[collectionIndex]
                = picongpu::particles::atomicPhysics2::rateCalculation ::multiplicityConfigNumber<ConfigNumber>(
                    configNumber);
        }

        /** returns collection index of atomic state in dataBox with given ConfigNumber
         *
         * @attention avoid use if possible in favor of direct access using collectionIndex
         *
         * @param configNumber ... configNumber of atomic state
         * @param startIndexBlock ... start index for search, not required but faster,
         *  is available from chargeStateOrgaDataBox.startIndexBlockAtomicStates(chargeState)
         *  with chargeState available from ConfigNumber::getIonizationState(configNumber)
         * @param numberAtomicStatesForChargeState ... number of atomic states in model with charge state
         *  of configNumber, not required but faster, use number atomic states if unknown,
         *  is available from chargeStateOrgaDataBox.numberAtomicStates(chargeState).
         *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
         *
         * @return returns numStates if not found
         */
        HDINLINE uint32_t findStateCollectionIndex(
            Idx const configNumber,
            uint32_t const numberAtomicStatesForChargeState,
            uint32_t const startIndexBlock = 0u) const
        {
            /// @todo replace linear search, BrianMarre, 2022
            // search for state in dataBox
            for(uint32_t i = 0; i < numberAtomicStatesForChargeState; ++i)
            {
                if(m_boxConfigNumber(i + startIndexBlock) == configNumber)
                {
                    return i + startIndexBlock;
                }
            }

            // atomic state not found return known bad value
            return m_numberAtomicStates;
        }

        /** returns collection index of atomic state in dataBox with lowest energy in specified index range
         *
         * @attention only use for search within a single charge state!, does not consider ionization energies, only
         * excitation energies
         * @attention assumes range contains at least one atomic state
         *
         * @param startIndexBlock start index for search,
         *  is available from chargeStateOrgaDataBox.startIndexBlockAtomicStates(chargeState)
         *  with chargeState available from ConfigNumber::getIonizationState(configNumber)
         * @param numberAtomicStatesForChargeState number of atomic states in model with charge state
         *  of configNumber, is available from chargeStateOrgaDataBox.numberAtomicStates(chargeState).
         *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
         */
        HDINLINE uint32_t
        findGroundState(uint32_t const numberAtomicStatesForChargeState, uint32_t const startIndexBlock) const
        {
            // get first state block energy
            float_X lowestEnergy = m_boxStateEnergy(startIndexBlock);
            uint32_t collectionIndex = startIndexBlock;

            /// @todo replace linear search, BrianMarre, 2023
            // search for state in dataBox
            for(uint32_t i = 1; i < numberAtomicStatesForChargeState; ++i)
            {
                if(m_boxStateEnergy(i + startIndexBlock) < lowestEnergy)
                {
                    lowestEnergy = m_boxStateEnergy(i + startIndexBlock);
                    collectionIndex = i + startIndexBlock;
                }
            }
            return collectionIndex;
        }

        /** returns the energy of the given state respective to the ground state of its ionization state
         *
         * @param ConfigNumber ... configNumber of atomic state
         * @param startIndexBlock ... start index for search, not required but faster,
         *  is available from chargeStateOrgaDataBox.startIndexBlockAtomicStates(chargeState)
         *  with chargeState available from ConfigNumber::getIonizationState(configNumber)
         * @param numberAtomicStatesForChargeState ... number of atomic states in model with charge state
         *  of configNumber, not required but faster, use number atomic states if unknown,
         *  is available from chargeStateOrgaDataBox.numberAtomicStates(chargeState).
         *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
         *
         * @return unit: eV
         */
        HDINLINE typename S_DataBox::TypeValue getEnergy(
            Idx const configNumber,
            uint32_t const numberAtomicStatesForChargeState,
            uint32_t const startIndexBlock = 0u) const
        {
            // special case completely ionized ion
            if(configNumber == 0u)
                return static_cast<typename S_DataBox::TypeValue>(0.0_X);

            uint32_t collectionIndex
                = findStateCollectionIndex(configNumber, startIndexBlock, numberAtomicStatesForChargeState);

            // atomic state not found, return zero, by definition isolated state
            if(collectionIndex == m_numberAtomicStates)
            {
                return static_cast<typename S_DataBox::TypeValue>(0._X);
            }

            // standard case
            return m_boxStateEnergy(collectionIndex);
        }

        /** returns configNumber of state corresponding to given index
         *
         * @attention no range check outside debug compile, invalid memory access if collectionIndex >=
         * numberAtomicStates
         * @param collectionIndex index of data box entry to query
         */
        HDINLINE Idx configNumber(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= m_numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of bounds atomic state configNumber call\n");
                    return static_cast<typename S_DataBox::TypeValue>(0._X);
                }

            return this->m_boxConfigNumber(collectionIndex);
        }

        /** directly query energy dataBox entry, use getEnergy() unless you know what you are doing!
         *
         * @attention no range check outside debug compile, invalid memory access if collectionIndex >=
         * numberAtomicStates
         * @param collectionIndex index of data box entry to query
         *
         * @return unit: eV
         */
        HDINLINE typename S_DataBox::TypeValue energy(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= m_numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of bounds atomic state energy call\n");
                    return static_cast<typename S_DataBox::TypeValue>(0._X);
                }

            return this->m_boxStateEnergy(collectionIndex);
        }

        /** returns multiplicity, number of physical states, of state corresponding to given index
         *
         * @attention no range check outside debug compile, invalid memory access if collectionIndex >=
         * numberAtomicStates
         * @param collectionIndex index of data box entry to query
         */
        HDINLINE TypeMultiplicity multiplicity(uint32_t const collectionIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(collectionIndex >= m_numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of bounds atomic state energy call\n");
                    return static_cast<typename S_DataBox::TypeValue>(0._X);
                }

            return this->m_boxMultiplicity(collectionIndex);
        }

        //! directly query get number of known atomic states
        HDINLINE uint32_t numberAtomicStatesTotal() const
        {
            return m_numberAtomicStates;
        }
    };

    /** complementing buffer class
     *
     * @tparam T_DataBoxType dataBox type used for storage
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_ConfigNumber dataType used for storage of configNumber of atomic states
     */
    template<typename T_Number, typename T_Value, typename T_ConfigNumber, typename T_Multiplicity>
    class AtomicStateDataBuffer : public DataBuffer<T_Number, T_Value>
    {
    public:
        using Idx = typename T_ConfigNumber::DataType;

        using TypeMultiplicity = T_Multiplicity;
        using BufferMultiplicity = pmacc::HostDeviceBuffer<TypeMultiplicity, 1u>;

        using ConfigNumber = T_ConfigNumber;
        using BufferConfigNumber = pmacc::HostDeviceBuffer<typename T_ConfigNumber::DataType, 1u>;

        using S_DataBuffer = DataBuffer<T_Number, T_Value>;

    private:
        std::unique_ptr<BufferConfigNumber> bufferConfigNumber;
        std::unique_ptr<typename S_DataBuffer::BufferValue> bufferStateEnergy;
        std::unique_ptr<BufferMultiplicity> bufferMultiplicity;

        uint32_t m_numberAtomicStates;

    public:
        HINLINE AtomicStateDataBuffer(uint32_t numberAtomicStates) : m_numberAtomicStates(numberAtomicStates)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutAtomicStates
                = pmacc::GridLayout<1>(numberAtomicStates, guardSize).getDataSpaceWithoutGuarding();

            bufferConfigNumber.reset(new BufferConfigNumber(layoutAtomicStates, false));
            bufferStateEnergy.reset(new typename S_DataBuffer::BufferValue(layoutAtomicStates, false));
            bufferMultiplicity.reset(new BufferMultiplicity(layoutAtomicStates, false));
        }

        HINLINE AtomicStateDataBox<T_Number, T_Value, T_ConfigNumber, T_Multiplicity> getHostDataBox()
        {
            return AtomicStateDataBox<T_Number, T_Value, T_ConfigNumber, T_Multiplicity>(
                bufferConfigNumber->getHostBuffer().getDataBox(),
                bufferStateEnergy->getHostBuffer().getDataBox(),
                bufferMultiplicity->getHostBuffer().getDataBox(),
                m_numberAtomicStates);
        }

        HINLINE AtomicStateDataBox<T_Number, T_Value, T_ConfigNumber, T_Multiplicity> getDeviceDataBox()
        {
            return AtomicStateDataBox<T_Number, T_Value, T_ConfigNumber, T_Multiplicity>(
                bufferConfigNumber->getDeviceBuffer().getDataBox(),
                bufferStateEnergy->getDeviceBuffer().getDataBox(),
                bufferMultiplicity->getDeviceBuffer().getDataBox(),
                m_numberAtomicStates);
        }

        //! get number of known atomic states
        HINLINE uint32_t getNumberAtomicStatesTotal() const
        {
            return m_numberAtomicStates;
        }

        HINLINE void hostToDevice()
        {
            bufferConfigNumber->hostToDevice();
            bufferStateEnergy->hostToDevice();
            bufferMultiplicity->hostToDevice();
        }

        HINLINE void deviceToHost()
        {
            bufferConfigNumber->deviceToHost();
            bufferStateEnergy->deviceToHost();
            bufferMultiplicity->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics2::atomicData
