/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov,
 *                     Brian Marre
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/param/physicalConstants.param"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <cstdint>
#include <memory>
#include <utility>

#pragma once

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            /** too different classes giving acess to atomic data:
             * - base class ... implements actual functionality
             * - dataBox class ... provides acess implementation for actual storage in box
             *      encapsulates index shift currently
             *
             * the atomic data box contains actually two data sets:
             *  - list of levels [(configNumber,
             *                    energy respective to ground state of ionzation state,
             *                    number of transitions,
             *                    startindex of block in transition list)]
             *  - list of transitions [(collisionalOscillatorStrength,
             *                          absorptionOscillatorStrength,
             *                          gaunt coefficent 1,
             *                          gaunt coefficent 2,
             *                          gaunt coefficent 3,
             *                          gaunt coefficent 4,
             *                          gaunt coefficent 5,
             *                          upper state configNumber)]
             *      transitions are grouped by lower state, the start of each block is
             *          specified in the state list
             *
             * NOTE: - configNumber specifies the number of a state as defined by the configNumber class
             *       - index always refers to a collection index
             *      the configNumber of a given state is always the same, its collection index depends on
             *      input file, => should only be used internally
             */

            template<
                uint8_t T_atomicNumber,
                // Data box types for atomic data on host and device
                typename T_DataBoxValue,
                typename T_DataBoxNumber,
                typename T_DataBoxStateConfigNumber,
                typename T_ConfigNumberDataType>
            class AtomicDataBox
            {
            public:
                using DataBoxValue = T_DataBoxValue;
                using DataBoxNumber = T_DataBoxNumber;
                using DataBoxStateConfigNumber = T_DataBoxStateConfigNumber;
                using Idx = T_ConfigNumberDataType;
                using ValueType = typename DataBoxValue::ValueType;

            private:
                DataBoxValue m_boxStateEnergy;
                DataBoxNumber m_boxNumTransitions;
                DataBoxNumber m_boxStartIndexBlockTransitions;
                DataBoxStateConfigNumber m_boxStateConfigNumber;
                uint32_t m_numStates;
                uint32_t m_maxNumberStates; // max number of States that can be stored

                DataBoxStateConfigNumber m_boxUpperConfigNumber; // lower config numebr is available via index
                DataBoxValue m_boxCollisionalOscillatorStrength;
                DataBoxValue m_boxCinx1;
                DataBoxValue m_boxCinx2;
                DataBoxValue m_boxCinx3;
                DataBoxValue m_boxCinx4;
                DataBoxValue m_boxCinx5;
                DataBoxValue m_boxAbsorptionOscillatorStrength;

                uint32_t m_numberTransitions;
                uint32_t m_maxNumberTransitions; // max number of Transitions that can be stored

            public:
                // Constructor
                AtomicDataBox(
                    DataBoxValue boxStateEnergy,
                    DataBoxNumber boxNumTransitions,
                    DataBoxNumber boxStartIndexBlockTransitions,
                    DataBoxStateConfigNumber boxStateConfigNumber,
                    uint32_t numStates,
                    uint32_t maxNumberStates,

                    DataBoxStateConfigNumber boxUpperConfigNumber,
                    DataBoxValue boxCollisionalOscillatorStrength,
                    DataBoxValue boxCinx1,
                    DataBoxValue boxCinx2,
                    DataBoxValue boxCinx3,
                    DataBoxValue boxCinx4,
                    DataBoxValue boxCinx5,
                    DataBoxValue boxAbsorptionOscillatorStrength,
                    uint32_t numberTransitions,
                    uint32_t maxNumberTransitions)
                    : m_boxStateEnergy(boxStateEnergy)
                    , m_boxNumTransitions(boxNumTransitions)
                    , m_boxStartIndexBlockTransitions(boxStartIndexBlockTransitions)
                    , m_boxStateConfigNumber(boxStateConfigNumber)
                    , m_numStates(numStates)
                    , m_maxNumberStates(maxNumberStates)

                    , m_boxUpperConfigNumber(boxUpperConfigNumber)
                    , m_boxCollisionalOscillatorStrength(boxCollisionalOscillatorStrength)
                    , m_boxCinx1(boxCinx1)
                    , m_boxCinx2(boxCinx2)
                    , m_boxCinx3(boxCinx3)
                    , m_boxCinx4(boxCinx4)
                    , m_boxCinx5(boxCinx5)
                    , m_boxAbsorptionOscillatorStrength(boxAbsorptionOscillatorStrength)
                    , m_numberTransitions(numberTransitions)
                    , m_maxNumberTransitions(maxNumberTransitions)
                {
                }

                /** write atomic data to console
                 *
                 * should be called serially
                 */
                HINLINE void writeToConsoleAtomicData() const
                {
                    std::cout << "(" << m_numStates << ", " << m_numberTransitions << ")" << std::endl;

                    for(uint32_t i = 0u; i < this->m_numStates; i++)
                    {
                        std::cout << i << " : "
                                  << "[" << m_boxStateConfigNumber(i) << "]" << m_boxStateEnergy(i) << " ("
                                  << m_boxNumTransitions(i) << "): " << m_boxStartIndexBlockTransitions(i)
                                  << std::endl;

                        for(uint32_t j = 0; j < m_boxNumTransitions(i); j++)
                        {
                            uint32_t indexTransition = m_boxStartIndexBlockTransitions(i) + j;

                            std::cout << "\t" << indexTransition << " : " << m_boxStateConfigNumber(i) << " -> "
                                      << m_boxUpperConfigNumber(indexTransition)
                                      << "; C: " << m_boxCollisionalOscillatorStrength(indexTransition)
                                      << ", A: " << m_boxAbsorptionOscillatorStrength(indexTransition) << ", Gaunt: ( "
                                      << m_boxCinx1(indexTransition) << ", " << m_boxCinx2(indexTransition) << ", "
                                      << m_boxCinx3(indexTransition) << ", " << m_boxCinx4(indexTransition) << ", "
                                      << m_boxCinx5(indexTransition) << ")" << std::endl;
                        }
                    }
                }

                /**returns the energy of the given state respective to the ground state of its ionization
                 *
                 * @param ConfigNumber ... configNumber of atomic state
                 * return unit: ATOMIC_UNIT_ENERGY
                 */
                // @TODO: replace dumb linear search @BrianMarre 2021
                HDINLINE ValueType operator()(Idx const ConfigNumber) const
                {
                    // one is a special case
                    if(ConfigNumber == 0)
                        return 0.0_X;

                    // search for state in list
                    for(uint32_t i = 0u; i < this->m_numStates; i++)
                    {
                        if(m_boxStateConfigNumber(i) == ConfigNumber)
                        {
                            // NOTE: unit conversion should be done in 64 bit
                            return float_X(float_64(m_boxStateEnergy(i)) * UNITCONV_eV_to_AU);
                        }
                    }
                    // atomic state not found return zero
                    return static_cast<ValueType>(0);
                }

                /** returns state corresponding to given index */
                HDINLINE Idx getAtomicStateConfigNumberIndex(uint32_t const indexState) const
                {
                    return this->m_boxStateConfigNumber(indexState);
                }

                /** returns index of atomic state in databox, if returns numStates state not found
                 *
                 * @TODO: replace linear search @BrianMarre, 2021
                 */
                HDINLINE uint32_t findState(Idx const stateConfigNumber) const
                {
                    for(uint32_t i = 0u; i < this->m_numStates; i++)
                    {
                        if(this->m_boxStateConfigNumber(i) == stateConfigNumber)
                            return i;
                    }
                    return m_numStates;
                }

                /** returns index of transition in databox, if retunrs numberTransitions not found
                 *
                 * @TODO: replace linear search
                 */
                HDINLINE uint32_t findTransition(Idx const lowerConfigNumber, Idx const upperConfigNumber) const
                {
                    // search for lowerConfigNumber in state list
                    for(uint32_t i = 0u; i < this->m_numStates; i++)
                        if(m_boxStateConfigNumber(i) == lowerConfigNumber)
                        {
                            // search in corresponding block in transitions box
                            for(uint32_t j = 0u; j < this->m_boxNumTransitions(i); j++)
                            {
                                // Does Lower state have at least one transition?,
                                // otherwise StartIndexBlockTransition == m_maxNumberTransitions
                                if((this->m_boxStartIndexBlockTransitions(i) < (this->m_maxNumberTransitions)) &&
                                   // is correct upperConfigNumber?
                                   this->m_boxUpperConfigNumber(this->m_boxStartIndexBlockTransitions(i) + j)
                                       == upperConfigNumber)
                                    return this->m_boxStartIndexBlockTransitions(i) + j;
                            }
                        }
                    return this->m_numberTransitions;
                }

                /** searches for transition to upper state in block of transitions of lower State,
                 *  returns index in databox of this transition if found, or m_numberTransitions if not
                 */
                HDINLINE uint32_t
                findTransitionInBlock(uint32_t const indexLowerState, Idx const upperConfigNumber) const
                {
                    uint32_t startIndexBlock = this->m_boxStartIndexBlockTransitions(indexLowerState);

                    for(uint32_t i = 0u; i < this->m_boxNumTransitions(indexLowerState); i++)
                    {
                        if(this->m_boxUpperConfigNumber(startIndexBlock + i) == upperConfigNumber)
                            return this->m_boxStartIndexBlockTransitions(indexLowerState) + i;
                    }
                    return this->m_numberTransitions;
                }

                /** returns upper states ConfigNumber of the transition
                 *
                 * @param indexTransition ... collection index of transition,
                 *  available using findTransition() and findTransitionInBlock()
                 */
                HDINLINE Idx getUpperConfigNumberTransition(uint32_t const indexTransition) const
                {
                    return this->m_boxUpperConfigNumber(indexTransition);
                }

                /** returns number of Transitions in dataBox with state as lower state
                 *
                 *  @param stateIndex ... collection index of state, available using findState()
                 */
                HDINLINE uint32_t getNumberTransitions(uint32_t const indexState) const
                {
                    if(indexState < m_numStates)
                        return this->m_boxNumTransitions(indexState);
                    return 0u;
                }

                // returns start index of the block of transitions with state as lower state
                HDINLINE uint32_t getStartIndexBlock(uint32_t const indexState) const
                {
                    return this->m_boxStartIndexBlockTransitions(indexState);
                }

                // number of Transitions stored in this box
                HDINLINE uint32_t getNumTransitions() const
                {
                    return this->m_numberTransitions;
                }

                // number of atomic states stored in this box
                HDINLINE uint32_t getNumStates() const
                {
                    return this->m_numStates;
                }

                HDINLINE ValueType getCollisionalOscillatorStrength(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCollisionalOscillatorStrength(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx1(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx1(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx2(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx2(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx3(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx3(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx4(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx4(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx5(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx5(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getAbsorptionOscillatorStrength(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxAbsorptionOscillatorStrength(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE constexpr static uint8_t getAtomicNumber()
                {
                    return T_atomicNumber;
                }


                HDINLINE void addLevel(
                    Idx const ConfigNumber, // must be index as defined in ConfigNumber
                    ValueType const energy // unit: eV
                )
                {
                    /** add a level to databox
                     *
                     *  NOTE: - must be called sequentially!
                     *        - assumes no more levels are added than memory is available,
                     */

                    if(this->m_numStates < m_maxNumberStates)
                    {
                        this->m_boxStateConfigNumber[this->m_numStates] = ConfigNumber;
                        this->m_boxStateEnergy[this->m_numStates] = energy;
                        this->m_boxNumTransitions[this->m_numStates] = 0u;
                        this->m_boxStartIndexBlockTransitions[this->m_numStates] = this->m_maxNumberTransitions;
                        this->m_numStates += 1u;
                    }
                }

                /** add transition to atomic data box
                 *
                 *  NOTE: must be called block wise and sequentially!
                 *  - block wise: add all transitions of one lower ConfigNumber before moving on
                 *      to the next lowerConfigNumber value
                 */
                HDINLINE void addTransition(
                    Idx const lowerConfigNumber, // must be index as defined in ConfigNumber
                    Idx const upperConfigNumber, // must be index as defined in ConfigNumber
                    ValueType const collisionalOscillatorStrength, // unit: unitless
                    ValueType const gauntCoefficent1, // unit: unitless
                    ValueType const gauntCoefficent2, // unit: unitless
                    ValueType const gauntCoefficent3, // unit: unitless
                    ValueType const gauntCoefficent4, // unit: unitless
                    ValueType const gauntCoefficent5, // unit: unitless
                    ValueType const absorptionOscillatorStrength) // unitless
                {
                    // get dataBox index of lowerConfigNumber
                    uint32_t collectionIndex = this->findState(lowerConfigNumber);

                    // check transition actually found
                    if(collectionIndex == this->m_numStates)
                    {
                        printf("ERROR: Tried adding transition without adding lower level first");
                        return;
                    }

                    // set start index block in transition collection if first transition of this lowerConfigNumber
                    if(this->m_boxStartIndexBlockTransitions(collectionIndex) == m_maxNumberTransitions)
                    {
                        this->m_boxStartIndexBlockTransitions(collectionIndex) = m_numberTransitions;
                    }

                    // check not too many transitions
                    if((this->m_numberTransitions < m_maxNumberTransitions))
                    {
                        // input transition data
                        this->m_boxUpperConfigNumber[m_numberTransitions] = upperConfigNumber;
                        this->m_boxCollisionalOscillatorStrength[m_numberTransitions] = collisionalOscillatorStrength;
                        this->m_boxCinx1[m_numberTransitions] = gauntCoefficent1;
                        this->m_boxCinx2[m_numberTransitions] = gauntCoefficent2;
                        this->m_boxCinx3[m_numberTransitions] = gauntCoefficent3;
                        this->m_boxCinx4[m_numberTransitions] = gauntCoefficent4;
                        this->m_boxCinx5[m_numberTransitions] = gauntCoefficent5;
                        this->m_boxAbsorptionOscillatorStrength[m_numberTransitions] = absorptionOscillatorStrength;

                        // update context
                        this->m_numberTransitions += 1u;
                        this->m_boxNumTransitions(collectionIndex) += 1u;
                    }
                }
            };


            // atomic data box host-device storage,
            // to be used from the host side only
            template<uint8_t T_atomicNumber, typename T_ConfigNumberDataType = uint64_t>
            class AtomicData
            {
            public:
                // type declarations
                using Idx = T_ConfigNumberDataType;
                using BufferValue = pmacc::GridBuffer<float_X, 1>;
                using BufferNumber = pmacc::GridBuffer<uint32_t, 1>;
                using BufferConfigNumber = pmacc::GridBuffer<T_ConfigNumberDataType, 1>;

                // data storage
                using InternalDataBoxTypeValue = pmacc::DataBox<pmacc::PitchedBox<float_X, 1>>;
                using InternalDataBoxTypeNumber = pmacc::DataBox<pmacc::PitchedBox<uint32_t, 1>>;
                using InternalDataBoxTypeConfigNumber = pmacc::DataBox<pmacc::PitchedBox<T_ConfigNumberDataType, 1>>;

                // acess datatype used on device
                using DataBoxType = AtomicDataBox<
                    T_atomicNumber,
                    InternalDataBoxTypeValue,
                    InternalDataBoxTypeNumber,
                    InternalDataBoxTypeConfigNumber,
                    T_ConfigNumberDataType>;

            private:
                // pointers to storage
                std::unique_ptr<BufferValue>
                    dataStateEnergy; // unit: eV, @TODO change to ATOMIC_UNIT_ENERGY?, BrianMarre, 2021
                std::unique_ptr<BufferNumber> dataNumTransitions; // unit: unitless
                std::unique_ptr<BufferNumber> dataStartIndexBlockTransitions; // unit: unitless
                std::unique_ptr<BufferConfigNumber> dataConfigNumber; // unit: unitless

                std::unique_ptr<BufferConfigNumber> dataUpperConfigNumber; // unit: unitless
                std::unique_ptr<BufferValue> dataCollisionalOscillatorStrength; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx1; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx2; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx3; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx4; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx5; // unit: unitless
                std::unique_ptr<BufferValue> dataAbsorptionOscillatorStrength; // unit: unitless

                // number of states included in atomic data
                uint32_t m_maxNumberStates;
                uint32_t m_maxNumberTransitions;

            public:
                HINLINE AtomicData(uint32_t maxNumberStates, uint32_t maxNumberTransitions)
                {
                    m_maxNumberStates = maxNumberStates;
                    m_maxNumberTransitions = maxNumberTransitions;

                    // get values for init of databox
                    auto sizeStates = pmacc::DataSpace<1>::create(m_maxNumberStates);
                    auto sizeTransitions = pmacc::DataSpace<1>::create(m_maxNumberTransitions);

                    auto const guardSize = pmacc::DataSpace<1>::create(0);

                    auto const layoutStates = pmacc::GridLayout<1>(sizeStates, guardSize);
                    auto const layoutTransitions = pmacc::GridLayout<1>(sizeTransitions, guardSize);

                    // create Buffers on stack and store pointer to it as member
                    // states data
                    dataConfigNumber.reset(new BufferConfigNumber(layoutStates));
                    dataStateEnergy.reset(new BufferValue(layoutStates));
                    dataNumTransitions.reset(new BufferNumber(layoutStates));
                    dataStartIndexBlockTransitions.reset(new BufferNumber(layoutStates));

                    // transition data
                    dataUpperConfigNumber.reset(new BufferConfigNumber(layoutTransitions));
                    dataCollisionalOscillatorStrength.reset(new BufferValue(layoutTransitions));
                    dataCinx1.reset(new BufferValue(layoutTransitions));
                    dataCinx2.reset(new BufferValue(layoutTransitions));
                    dataCinx3.reset(new BufferValue(layoutTransitions));
                    dataCinx4.reset(new BufferValue(layoutTransitions));
                    dataCinx5.reset(new BufferValue(layoutTransitions));
                    dataAbsorptionOscillatorStrength.reset(new BufferValue(layoutTransitions));
                }

                //! Get the host data box for the rate matrix values
                HINLINE DataBoxType getHostDataBox(uint32_t numStates, uint32_t numberTransitions)
                {
                    return DataBoxType(
                        dataStateEnergy->getHostBuffer().getDataBox(),
                        dataNumTransitions->getHostBuffer().getDataBox(),
                        dataStartIndexBlockTransitions->getHostBuffer().getDataBox(),
                        dataConfigNumber->getHostBuffer().getDataBox(),
                        numStates,
                        this->m_maxNumberStates,

                        // dataLowerConfigNumber->getHostBuffer().getDataBox(),
                        dataUpperConfigNumber->getHostBuffer().getDataBox(),
                        dataCollisionalOscillatorStrength->getHostBuffer().getDataBox(),
                        dataCinx1->getHostBuffer().getDataBox(),
                        dataCinx2->getHostBuffer().getDataBox(),
                        dataCinx3->getHostBuffer().getDataBox(),
                        dataCinx4->getHostBuffer().getDataBox(),
                        dataCinx5->getHostBuffer().getDataBox(),
                        dataAbsorptionOscillatorStrength->getHostBuffer().getDataBox(),
                        numberTransitions,
                        this->m_maxNumberTransitions);
                }

                //! Get the device data box for the rate matrix values
                HINLINE DataBoxType getDeviceDataBox(uint32_t numStates, uint32_t numberTransitions)
                {
                    return DataBoxType(
                        dataStateEnergy->getDeviceBuffer().getDataBox(),
                        dataNumTransitions->getDeviceBuffer().getDataBox(),
                        dataStartIndexBlockTransitions->getDeviceBuffer().getDataBox(),
                        dataConfigNumber->getDeviceBuffer().getDataBox(),
                        numStates,
                        this->m_maxNumberStates,

                        dataUpperConfigNumber->getDeviceBuffer().getDataBox(),
                        dataCollisionalOscillatorStrength->getDeviceBuffer().getDataBox(),
                        dataCinx1->getDeviceBuffer().getDataBox(),
                        dataCinx2->getDeviceBuffer().getDataBox(),
                        dataCinx3->getDeviceBuffer().getDataBox(),
                        dataCinx4->getDeviceBuffer().getDataBox(),
                        dataCinx5->getDeviceBuffer().getDataBox(),
                        dataAbsorptionOscillatorStrength->getDeviceBuffer().getDataBox(),
                        numberTransitions,
                        this->m_maxNumberTransitions);
                }

                void syncToDevice()
                {
                    dataStateEnergy->hostToDevice();
                    dataNumTransitions->hostToDevice();
                    dataStartIndexBlockTransitions->hostToDevice();
                    dataConfigNumber->hostToDevice();

                    dataUpperConfigNumber->hostToDevice();
                    dataCollisionalOscillatorStrength->hostToDevice();
                    dataCinx1->hostToDevice();
                    dataCinx2->hostToDevice();
                    dataCinx3->hostToDevice();
                    dataCinx4->hostToDevice();
                    dataCinx5->hostToDevice();
                    dataAbsorptionOscillatorStrength->hostToDevice();
                }
            };

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
