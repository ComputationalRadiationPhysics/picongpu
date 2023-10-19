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

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/static_assert.hpp>

// charge state data
#include "picongpu/particles/atomicPhysics2/atomicData/ChargeStateData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/ChargeStateOrgaData.hpp"
// atomic state data
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateNumberOfTransitionsData_Down.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateNumberOfTransitionsData_UpDown.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateStartIndexBlockData_Down.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateStartIndexBlockData_UpDown.hpp"
// transition data
#include "picongpu/particles/atomicPhysics2/atomicData/AutonomousTransitionData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/BoundBoundTransitionData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/BoundFreeTransitionData.hpp"
// precomputed cache for chooseTransition kernel
#include "picongpu/particles/atomicPhysics2/atomicData/TransitionSelectionData.hpp"

// tuple definitions
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"
// helper stuff for transition tuples
#include "picongpu/particles/atomicPhysics2/atomicData/CheckTransitionTuple.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/CompareTransitionTuple.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/GetStateFromTransitionTuple.hpp"

// enum of groups of processClass's
#include "picongpu/particles/atomicPhysics2/processClass/ProcessClassGroup.hpp"
// enum of transition ordering in dataBoxes
#include "picongpu/particles/atomicPhysics2/processClass/TransitionOrdering.hpp"

// number of physical transitions for each transition data entry
#include "picongpu/particles/atomicPhysics2/ConvertEnumToUint.hpp"
#include "picongpu/particles/atomicPhysics2/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics2/processClass/NumberPhysicalTransitions.hpp"

// debug only
#include "picongpu/particles/atomicPhysics2/DebugHelper.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

//! @file gathers atomic data storage implementations and implements filling them on runtime
namespace picongpu::particles::atomicPhysics2::atomicData
{
    namespace procClass = picongpu::particles::atomicPhysics2::processClass;
    using ProcClassGroup = picongpu::particles::atomicPhysics2::processClass::ProcessClassGroup;

    /** gathering of all atomicPhyiscs input data
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_CollectionIndex dataType used for collection index, typically uint32_t
     * @tparam T_ConfigNumber dataType used for storage of configNumber
     * @tparam T_Multiplicities dataType used for storage of T_Multiplicities, typically uint64_t
     *
     * @tparam T_electronicExcitation is channel active?
     * @tparam T_electronicDeexcitation is channel active?
     * @tparam T_spontaneousDeexcitation is channel active?
     * @tparam T_autonomousIonization is channel active?
     * @tparam T_electronicIonization is channel active?
     * @tparam T_fieldIonization is channel active?
     *
     * The atomicPhysics step relies on a model of atomic states and transitions for each
     * atomicPhysics ion species.
     * These model's parameters are provided by the user as a .txt file of specified format
     * (see documentation) at runtime.
     *
     *  PIConGPU itself only includes charge state data, for ADK-, Thomas-Fermi- and BSI-ionization.
     *  All other atomic state data is kept separate from PIConGPU itself, due to license requirements.
     *
     * This file is read at the start of the simulation and stored distributed over the following:
     *  - charge state property data [ChargeStateDataBox.hpp]
     *      * ionization energy
     *      * screened charge
     *  - charge state orga data [ChargeStateOrgaDataBox.hpp]
     *      * number of atomic states for each charge state
     *      * start index block for charge state in list of atomic states
     * - atomic state property data [AtomicStateDataBox.hpp]
     *      * configNumber
     *      * state energy, above ground state of charge state
     * - atomic state orga data
     *      [AtomicStateNumberOfTransitionsDataBox_Down, AtomicStateNumberOfTransitionsDataBox_UpDown]
     *       * number of transitions (up-/)down for each atomic state,
     *          by type of transition(bound-bound/bound-free/autonomous)
     *       * offset in transition selection ordering for each atomic state
     *      [AtomicStateStartIndexBlockDataBox_Down, AtomicStateStartIndexBlockDataBox_UpDown]
     *       * start index of block in transition collection index for atomic state,
     *          by type of transition(bound-bound/bound-free/autonomous)
     * - transition property data[BoundBoundTransitionDataBox, BoundFreeTransitionDataBox,
     *      AutonomousTransitionDataBox]
     *      * parameters for cross section calculation for each modelled transition
     *
     * (orga data describes the structure of the property data for faster lookups, lookups are
     *  are always possible without it, but are possibly non performant)
     *
     * For each of data subsets exists a dataBox class, a container class holding
     *      pmacc::dataBox'es, and a dataBuffer class, a container class holding
     *      pmacc::buffers in turn allowing access to the pmacc::dataBox'es.
     *
     * Each dataBuffer will create on demand a host or device side dataBox class object which
     *  in turn gives access to the data.
     *
     * Assumptions about input data are described in CheckTransitionTuple.hpp, ordering requirements of transitions in
     *  CompareTransitionTuple.hpp and all other requirements in the checkChargeStateList(), checkAtomicStateList() and
     *  checkTransitionsForEnergyInversion() methods.
     */
    template<
        typename T_Number,
        typename T_Value,
        typename T_CollectionIndex,
        typename T_ConfigNumber,
        typename T_Multiplicities,
        bool T_electronicExcitation,
        bool T_electronicDeexcitation,
        bool T_spontaneousDeexcitation,
        bool T_electronicIonization,
        bool T_autonomousIonization,
        bool T_fieldIonization> /// @todo add photonic channels, Brian Marre, 2022
    class AtomicData : public pmacc::ISimulationData
    {
    public:
        using TypeNumber = T_Number;
        using TypeValue = T_Value;
        using TypeCollectionIndex = T_CollectionIndex;
        using Idx = typename T_ConfigNumber::DataType;
        using ConfigNumber = T_ConfigNumber;

        static constexpr bool switchElectronicExcitation = T_electronicExcitation;
        static constexpr bool switchElectronicDeexcitation = T_electronicDeexcitation;
        static constexpr bool switchSpontaneousDeexcitation = T_spontaneousDeexcitation;
        static constexpr bool switchElectronicIonization = T_electronicIonization;
        static constexpr bool switchAutonomousIonization = T_autonomousIonization;
        static constexpr bool switchFieldIonization = T_fieldIonization;

        // tuples: S_* for shortened name
        using S_ChargeStateTuple = ChargeStateTuple<TypeValue>;
        using S_AtomicStateTuple = AtomicStateTuple<TypeValue, Idx>;
        using S_BoundBoundTransitionTuple = BoundBoundTransitionTuple<TypeValue, Idx>;
        using S_BoundFreeTransitionTuple = BoundFreeTransitionTuple<TypeValue, Idx>;
        using S_AutonomousTransitionTuple = AutonomousTransitionTuple<Idx>;

        // dataBuffers: S_* for shortened name
        using S_ChargeStateDataBuffer = ChargeStateDataBuffer<TypeNumber, TypeValue, T_ConfigNumber::atomicNumber>;
        using S_ChargeStateOrgaDataBuffer
            = ChargeStateOrgaDataBuffer<TypeNumber, TypeValue, T_ConfigNumber::atomicNumber>;
        using S_AtomicStateDataBuffer = AtomicStateDataBuffer<TypeNumber, TypeValue, T_ConfigNumber, T_Multiplicities>;

        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateStartIndexBlockDataBuffer_UpDown
            = AtomicStateStartIndexBlockDataBuffer_UpDown<TypeCollectionIndex, TypeValue, T_ProcessClassGroup>;
        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateStartIndexBlockDataBuffer_Down
            = AtomicStateStartIndexBlockDataBuffer_Down<TypeCollectionIndex, TypeValue, T_ProcessClassGroup>;
        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateNumberOfTransitionsDataBuffer_UpDown
            = AtomicStateNumberOfTransitionsDataBuffer_UpDown<TypeNumber, TypeValue, T_ProcessClassGroup>;
        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateNumberOfTransitionsDataBuffer_Down
            = AtomicStateNumberOfTransitionsDataBuffer_Down<TypeNumber, TypeValue, T_ProcessClassGroup>;

        template<procClass::TransitionOrdering T_TransitionOrdering>
        using S_BoundBoundTransitionDataBuffer = BoundBoundTransitionDataBuffer<
            TypeNumber,
            TypeValue,
            T_CollectionIndex,
            typename T_ConfigNumber::DataType,
            T_TransitionOrdering>;
        template<procClass::TransitionOrdering T_TransitionOrdering>
        using S_BoundFreeTransitionDataBuffer = BoundFreeTransitionDataBuffer<
            TypeNumber,
            TypeValue,
            T_CollectionIndex,
            T_ConfigNumber,
            T_Multiplicities,
            T_TransitionOrdering>;
        template<procClass::TransitionOrdering T_TransitionOrdering>
        using S_AutonomousTransitionDataBuffer = AutonomousTransitionDataBuffer<
            TypeNumber,
            TypeValue,
            T_CollectionIndex,
            typename T_ConfigNumber::DataType,
            T_TransitionOrdering>;

        using S_TransitionSelectionDataBuffer = TransitionSelectionDataBuffer<TypeNumber, TypeValue>;

        // dataBoxes: S_* for shortened name
        using S_ChargeStateDataBox = ChargeStateDataBox<TypeNumber, TypeValue, T_ConfigNumber::atomicNumber>;
        using S_ChargeStateOrgaDataBox = ChargeStateOrgaDataBox<TypeNumber, TypeValue, T_ConfigNumber::atomicNumber>;

        using S_AtomicStateDataBox = AtomicStateDataBox<TypeNumber, TypeValue, T_ConfigNumber, T_Multiplicities>;

        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateStartIndexBlockDataBox_UpDown
            = AtomicStateStartIndexBlockDataBox_UpDown<TypeNumber, TypeValue, T_ProcessClassGroup>;

        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateStartIndexBlockDataBox_Down
            = AtomicStateStartIndexBlockDataBox_Down<TypeNumber, TypeValue, T_ProcessClassGroup>;

        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateNumberOfTransitionsDataBox_UpDown
            = AtomicStateNumberOfTransitionsDataBox_UpDown<TypeNumber, TypeValue, T_ProcessClassGroup>;

        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateNumberOfTransitionsDataBox_Down
            = AtomicStateNumberOfTransitionsDataBox_Down<TypeNumber, TypeValue, T_ProcessClassGroup>;

        template<picongpu::particles::atomicPhysics2::processClass::TransitionOrdering T_TransitionOrdering>
        using S_BoundBoundTransitionDataBox = BoundBoundTransitionDataBox<
            TypeNumber,
            TypeValue,
            T_CollectionIndex,
            typename T_ConfigNumber::DataType,
            T_TransitionOrdering>;
        template<picongpu::particles::atomicPhysics2::processClass::TransitionOrdering T_TransitionOrdering>
        using S_BoundFreeTransitionDataBox = BoundFreeTransitionDataBox<
            TypeNumber,
            TypeValue,
            T_CollectionIndex,
            T_ConfigNumber,
            T_Multiplicities,
            T_TransitionOrdering>;
        template<picongpu::particles::atomicPhysics2::processClass::TransitionOrdering T_TransitionOrdering>
        using S_AutonomousTransitionDataBox = AutonomousTransitionDataBox<
            TypeNumber,
            TypeValue,
            T_CollectionIndex,
            typename T_ConfigNumber::DataType,
            T_TransitionOrdering>;

        using S_TransitionSelectionDataBox = TransitionSelectionDataBox<TypeNumber, TypeValue>;

    private:
        // pointers to storage
        // charge state data
        std::unique_ptr<S_ChargeStateDataBuffer> chargeStateDataBuffer;
        std::unique_ptr<S_ChargeStateOrgaDataBuffer> chargeStateOrgaDataBuffer;

        // atomic property data
        std::unique_ptr<S_AtomicStateDataBuffer> atomicStateDataBuffer;
        // atomic orga data
        std::unique_ptr<S_AtomicStateStartIndexBlockDataBuffer_UpDown<ProcClassGroup::boundBoundBased>>
            atomicStateStartIndexBlockDataBuffer_BoundBound;
        std::unique_ptr<S_AtomicStateStartIndexBlockDataBuffer_UpDown<ProcClassGroup::boundFreeBased>>
            atomicStateStartIndexBlockDataBuffer_BoundFree;
        std::unique_ptr<S_AtomicStateStartIndexBlockDataBuffer_Down<ProcClassGroup::autonomousBased>>
            atomicStateStartIndexBlockDataBuffer_Autonomous;
        std::unique_ptr<S_AtomicStateNumberOfTransitionsDataBuffer_UpDown<ProcClassGroup::boundBoundBased>>
            atomicStateNumberOfTransitionsDataBuffer_BoundBound;
        std::unique_ptr<S_AtomicStateNumberOfTransitionsDataBuffer_UpDown<ProcClassGroup::boundFreeBased>>
            atomicStateNumberOfTransitionsDataBuffer_BoundFree;
        std::unique_ptr<S_AtomicStateNumberOfTransitionsDataBuffer_Down<ProcClassGroup::autonomousBased>>
            atomicStateNumberOfTransitionsDataBuffer_Autonomous;

        // transition data, normal, sorted by lower state
        std::unique_ptr<S_BoundBoundTransitionDataBuffer<procClass::TransitionOrdering::byLowerState>>
            boundBoundTransitionDataBuffer;
        std::unique_ptr<S_BoundFreeTransitionDataBuffer<procClass::TransitionOrdering::byLowerState>>
            boundFreeTransitionDataBuffer;
        std::unique_ptr<S_AutonomousTransitionDataBuffer<procClass::TransitionOrdering::byLowerState>>
            autonomousTransitionDataBuffer;

        // transition data, inverted,sorted by upper state
        std::unique_ptr<S_BoundBoundTransitionDataBuffer<procClass::TransitionOrdering::byUpperState>>
            inverseBoundBoundTransitionDataBuffer;
        std::unique_ptr<S_BoundFreeTransitionDataBuffer<procClass::TransitionOrdering::byUpperState>>
            inverseBoundFreeTransitionDataBuffer;
        std::unique_ptr<S_AutonomousTransitionDataBuffer<procClass::TransitionOrdering::byUpperState>>
            inverseAutonomousTransitionDataBuffer;

        // transition selection data
        std::unique_ptr<S_TransitionSelectionDataBuffer> transitionSelectionDataBuffer;

        uint32_t m_numberAtomicStates = 0u;

        uint32_t m_numberBoundBoundTransitions = 0u;
        uint32_t m_numberBoundFreeTransitions = 0u;
        uint32_t m_numberAutonomousTransitions = 0u;

        const std::string m_speciesName;

        //! open file
        HINLINE static std::ifstream openFile(std::string const fileName, std::string const fileContent)
        {
            std::ifstream file(fileName);

            // check for success
            if(!file)
            {
                throw std::runtime_error("atomicPhysics ERROR: could not open " + fileContent + ": " + fileName);
            }

            return file;
        }

        /** read charge state data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - charge state data is sorted by ascending charge
         *   - the completely ionized state is left out
         *
         * @return returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_ChargeStateTuple> readChargeStates(std::string const fileName)
        {
            std::ifstream file = openFile(fileName, "charge state data");
            if(!file)
                return std::list<S_ChargeStateTuple>{};

            std::list<S_ChargeStateTuple> chargeStateList;

            TypeValue ionizationEnergy;
            TypeValue screenedCharge;
            uint32_t chargeState;
            uint8_t numberChargeStates = 0u;

            while(file >> chargeState >> ionizationEnergy >> screenedCharge)
            {
                if(chargeState == static_cast<uint32_t>(T_ConfigNumber::atomicNumber))
                    throw std::runtime_error(
                        "charge state " + std::to_string(chargeState)
                        + " should not be included in input file for Z = "
                        + std::to_string(T_ConfigNumber::atomicNumber));

                S_ChargeStateTuple item = std::make_tuple(
                    static_cast<uint8_t>(chargeState),
                    ionizationEnergy, // [eV]
                    screenedCharge); // [e]

                chargeStateList.push_back(item);

                numberChargeStates++;
            }

            if(numberChargeStates > T_ConfigNumber::atomicNumber)
                throw std::runtime_error(
                    "atomicPhysics ERROR: too many charge states, num > Z: "
                    + std::to_string(T_ConfigNumber::atomicNumber));

            return chargeStateList;
        }

        /** read atomic state data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - atomic state data is sorted block wise by charge state and secondary by ascending
         * configNumber
         *   - the completely ionized state is left out
         *
         * @return returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_AtomicStateTuple> readAtomicStates(std::string const fileName)
        {
            std::ifstream file = openFile(fileName, "atomic state data");
            if(!file)
                return std::list<S_AtomicStateTuple>{};

            std::list<S_AtomicStateTuple> atomicStateList;

            double stateConfigNumber;
            TypeValue energyOverGround;

            while(file >> stateConfigNumber >> energyOverGround)
            {
                S_AtomicStateTuple item = std::make_tuple(
                    static_cast<Idx>(stateConfigNumber), // unitless
                    energyOverGround); // [eV]

                atomicStateList.push_back(item);

                m_numberAtomicStates++;
            }

            return atomicStateList;
        }

        /** read bound-bound transitions data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - transition data is sorted block wise by lower atomic state and secondary by ascending by upper state
         * configNumber
         *
         * @return returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_BoundBoundTransitionTuple> readBoundBoundTransitions(std::string const fileName)
        {
            std::ifstream file = openFile(fileName, "bound-bound transition data");
            if(!file)
                return std::list<S_BoundBoundTransitionTuple>{};

            std::list<S_BoundBoundTransitionTuple> boundBoundTransitions;

            uint64_t idxLower;
            uint64_t idxUpper;
            TypeValue collisionalOscillatorStrength;
            TypeValue absorptionOscillatorStrength;

            // gauntCoeficients
            TypeValue cxin1;
            TypeValue cxin2;
            TypeValue cxin3;
            TypeValue cxin4;
            TypeValue cxin5;

            while(file >> idxLower >> idxUpper >> collisionalOscillatorStrength >> absorptionOscillatorStrength
                  >> cxin1 >> cxin2 >> cxin3 >> cxin4 >> cxin5)
            {
                Idx stateLower = static_cast<Idx>(idxLower);
                Idx stateUpper = static_cast<Idx>(idxUpper);

                // protection against circle transitions
                if(stateLower == stateUpper)
                {
                    std::cout << "atomicPhysics ERROR: circular transitions are not supported,"
                                 "treat steps separately"
                              << std::endl;
                    continue;
                }

                //

                S_BoundBoundTransitionTuple item = std::make_tuple(
                    collisionalOscillatorStrength,
                    absorptionOscillatorStrength,
                    cxin1,
                    cxin2,
                    cxin3,
                    cxin4,
                    cxin5,
                    stateLower,
                    stateUpper);

                boundBoundTransitions.push_back(item);
                m_numberBoundBoundTransitions++;
            }

            return boundBoundTransitions;
        }

        /** read bound-free transitions data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - transition data is sorted block wise by lower atomic state and secondary by ascending by upper state
         * configNumber
         *
         * @return returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_BoundFreeTransitionTuple> readBoundFreeTransitions(std::string const fileName)
        {
            std::ifstream file = openFile(fileName, "bound-free transition data");
            if(!file)
                return std::list<S_BoundFreeTransitionTuple>{};

            std::list<S_BoundFreeTransitionTuple> boundFreeTransitions;

            uint64_t idxLower;
            uint64_t idxUpper;

            // gauntCoeficients
            TypeValue cxin1;
            TypeValue cxin2;
            TypeValue cxin3;
            TypeValue cxin4;
            TypeValue cxin5;
            TypeValue cxin6;
            TypeValue cxin7;
            TypeValue cxin8;

            while(file >> idxLower >> idxUpper >> cxin1 >> cxin2 >> cxin3 >> cxin4 >> cxin5 >> cxin6 >> cxin7 >> cxin8)
            {
                Idx stateLower = static_cast<Idx>(idxLower);
                Idx stateUpper = static_cast<Idx>(idxUpper);

                // protection against circle transitions
                if(stateLower == stateUpper)
                {
                    std::cout << "atomicPhysics ERROR: circular transitions are not supported,"
                                 "treat steps separately"
                              << std::endl;
                    continue;
                }

                S_BoundFreeTransitionTuple item
                    = std::make_tuple(cxin1, cxin2, cxin3, cxin4, cxin5, cxin6, cxin7, cxin8, stateLower, stateUpper);

                boundFreeTransitions.push_back(item);
                m_numberBoundFreeTransitions++;
            }
            return boundFreeTransitions;
        }

        /** read autonomous transitions data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - transition data is sorted block wise by lower atomic state and secondary by ascending by upper state
         * configNumber
         *
         * @return returns empty list if file not found/(not accessible)
         */
        ALPAKA_FN_HOST std::list<S_AutonomousTransitionTuple> readAutonomousTransitions(std::string const fileName)
        {
            std::ifstream file = openFile(fileName, "autonomous transition data");
            if(!file)
                return std::list<S_AutonomousTransitionTuple>{};

            std::list<S_AutonomousTransitionTuple> autonomousTransitions;

            uint64_t idxLower;
            uint64_t idxUpper;

            // unit: 1/s
            float_64 rate;

            while(file >> idxLower >> idxUpper >> rate)
            {
                Idx stateLower = static_cast<Idx>(idxLower);
                Idx stateUpper = static_cast<Idx>(idxUpper);

                // protection against circle transitions
                if(stateLower == stateUpper)
                {
                    std::cout << "atomicPhysics ERROR: circular transitions are not supported,"
                                 "treat steps separately"
                              << std::endl;
                    continue;
                }

                const S_AutonomousTransitionTuple item = std::make_tuple(rate, stateLower, stateUpper);

                autonomousTransitions.push_back(item);
                m_numberAutonomousTransitions++;
            }
            return autonomousTransitions;
        }

        /** check charge state list
         *
         * @return passes silently if correct
         *
         * @throws runtime error if duplicate charge state, missing charge state,
         *  order broken(ascending in charge state), completely ionized state included or
         *  unphysical(Q > Z) charge state
         */
        ALPAKA_FN_HOST void checkChargeStateList(std::list<S_ChargeStateTuple>& chargeStateList)
        {
            typename std::list<S_ChargeStateTuple>::iterator iter = chargeStateList.begin();

            // running index of reads, i.e. expected charge state
            uint8_t chargeState = 1u;

            uint8_t lastChargeState;
            uint8_t currentChargeState;

            if(iter == chargeStateList.end())
                throw std::runtime_error("atomicPhysics ERROR: empty charge state list");

            lastChargeState = std::get<0>(*iter);
            iter++;

            if(lastChargeState != 0u)
                throw std::runtime_error("atomicPhysics ERROR: charge state 0 not first charge state");

            for(; iter != chargeStateList.end(); iter++)
            {
                currentChargeState = std::get<0>(*iter);

                // duplicate atomic state
                if(currentChargeState == lastChargeState)
                    throw std::runtime_error("atomicPhysics ERROR: duplicate charge state");

                // ordering
                if(!(currentChargeState > lastChargeState))
                    throw std::runtime_error("atomicPhysics ERROR: charge state ordering wrong");

                // missing charge state
                if(!(currentChargeState == chargeState))
                    throw std::runtime_error("atomicPhysics ERROR: charge state missing");

                // completely ionized state
                if(chargeState == T_ConfigNumber::atomicNumber)
                    throw std::runtime_error(
                        "atomicPhysics ERROR: completely ionized charge state found in charge state input data");

                // unphysical state
                if(chargeState > T_ConfigNumber::atomicNumber)
                    throw std::runtime_error("atomicPhysics ERROR: unphysical charge state found");

                chargeState++;
                lastChargeState = currentChargeState;
            }
        }

        /** check atomic state list
         *
         * @return passes silently if correct
         *
         * @throws runtime error if duplicate atomic state, primary order broken
         *  (ascending in charge state), secondary order broken(ascending in configNumber),
         *  or unphysical charge state found
         */
        ALPAKA_FN_HOST void checkAtomicStateList(std::list<S_AtomicStateTuple>& atomicStateList)
        {
            typename std::list<S_AtomicStateTuple>::iterator iter = atomicStateList.begin();

            Idx currentAtomicStateConfigNumber;
            uint8_t currentChargeState;

            // empty atomic state list, allowed for ground state only sims
            if(iter == atomicStateList.end())
                return;

            Idx lastAtomicStateConfigNumber = std::get<0>(*iter);
            uint8_t lastChargeState = ConfigNumber::getChargeState(lastAtomicStateConfigNumber);

            iter++;

            for(; iter != atomicStateList.end(); iter++)
            {
                currentAtomicStateConfigNumber = std::get<0>(*iter);
                currentChargeState = ConfigNumber::getChargeState(currentAtomicStateConfigNumber);

                // duplicate atomic state
                if(currentAtomicStateConfigNumber == lastAtomicStateConfigNumber)
                    throw std::runtime_error("atomicPhysics ERROR: duplicate atomic state");
                // later duplicate will break ordering

                // primary/secondary order
                if(currentChargeState == lastChargeState)
                {
                    // same block
                    if(currentAtomicStateConfigNumber < lastAtomicStateConfigNumber)
                        throw std::runtime_error(
                            "atomicPhysics ERROR: wrong secondary ordering of atomic states "
                            "(ascending in atomicConfigNumber), "
                            + std::to_string(currentAtomicStateConfigNumber) + " < "
                            + std::to_string(lastAtomicStateConfigNumber));
                }
                else
                {
                    // next block
                    if(currentChargeState < lastChargeState)
                        throw std::runtime_error(
                            "atomicPhysics ERROR: wrong primary ordering of atomic state, "
                            "(ascending in chargeState), "
                            + std::to_string(currentChargeState) + " < " + std::to_string(lastChargeState));
                }

                // completely ionized atomic state is allowed as upper state

                // unphysical atomic state
                if(currentChargeState > T_ConfigNumber::atomicNumber)
                    throw std::runtime_error("atomicPhysics ERROR: unphysical charge state found");

                lastChargeState = currentChargeState;
                lastAtomicStateConfigNumber = currentAtomicStateConfigNumber;
            }
        }

        /** check transition list
         *
         * @param transitionList
         *
         * @attention assumes that chargeStateList and atomicStateList fulfil all ordering assumptions
         * @return passes silently if correct
         * @throws runtime error if transition order broken, as defined by CompareTransitionTupel.hpp,
         *  or the transition tuple is not internally consistent(for its transition type),
         *  as defined by CheckTransitionTuple.hpp
         */
        template<typename T_TransitionTuple>
        ALPAKA_FN_HOST void checkTransitionList(std::list<T_TransitionTuple>& transitionList)
        {
            typename std::list<T_TransitionTuple>::iterator iter = transitionList.begin();

            // empty transition list, i.e. ground-ground transitions only
            if(iter == transitionList.end())
                return;

            // read first list entry as comparison point
            T_TransitionTuple lastTransitionTuple = *iter;
            checkTransitionTuple<ConfigNumber>(lastTransitionTuple);
            iter++;

            // read the rest
            T_TransitionTuple currentTransitionTuple;
            for(; iter != transitionList.end(); iter++)
            {
                currentTransitionTuple = *iter;

                // check transition tuple for internal consistency, charge states only
                checkTransitionTuple<ConfigNumber>(currentTransitionTuple);

                // check ordering, (lastTransitionTuple >= currentTransitionTuple)
                if(CompareTransitionTupel<TypeValue, ConfigNumber, /*order by Lower state*/ true>{}(
                       currentTransitionTuple,
                       lastTransitionTuple))
                {
                    // print debug info
                    picongpu::particles::atomicPhysics2::debug::
                        printTransitionTupleToConsole<T_TransitionTuple, Idx, TypeValue, ConfigNumber>(
                            lastTransitionTuple);
                    picongpu::particles::atomicPhysics2::debug::
                        printTransitionTupleToConsole<T_TransitionTuple, Idx, TypeValue, ConfigNumber>(
                            currentTransitionTuple);

                    throw std::runtime_error("atomicPhysics ERROR: wrong ordering of input data");
                }

                // move to next entry for comparison
                lastTransitionTuple = currentTransitionTuple;
            }
        }

        /** check that for all transitions in transitionHostBox, the lower state is lower in energy than the upper
         * state
         *
         * @param transitionHostBox host dataBox of transition data to check
         *
         * @attention assumes that all transition buffers have been filled previously
         */
        template<typename T_TransitionHostBox>
        ALPAKA_FN_HOST void checkTransitionsForEnergyInversion(T_TransitionHostBox transitionHostBox)
        {
            // for bound-free transitions upper- and lower-State are defined by charge state only!
            PMACC_CASSERT_MSG(
                wrong_or_unknown_transitionType_in_Energy_InversionCheck,
                ((u8(T_TransitionHostBox::processClassGroup) == u8(procClass::ProcessClassGroup::boundBoundBased))
                 || (u8(T_TransitionHostBox::processClassGroup)
                     == u8(procClass::ProcessClassGroup::autonomousBased))));

            uint32_t const numberTransitions = transitionHostBox.getNumberOfTransitionsTotal();

            for(uint32_t collectionIndex = static_cast<uint32_t>(0u); collectionIndex < numberTransitions;
                collectionIndex++)
            {
                float_X const deltaEnergy = picongpu::particles::atomicPhysics2::DeltaEnergyTransition::get(
                    collectionIndex,
                    this->getAtomicStateDataDataBox<true>(),
                    transitionHostBox,
                    this->getChargeStateDataDataBox<true>());

                if(deltaEnergy < 0._X)
                    throw std::runtime_error(
                        "atomicPhysics ERROR: upper and lower state inverted in energy in input"
                        ", energy lower state must be <= energy upper state, transitionType: "
                        + enumToString<T_TransitionHostBox::processClassGroup>() + " ,transition #"
                        + std::to_string(collectionIndex));
            }
        }

        /** init buffers,
         *
         * @attention all readMethods must have been executed exactly once before calling this method!,
         *  otherwise buffer size is unknown, depends in number of states/transitions in buffer
         */
        ALPAKA_FN_HOST void initBuffers()
        {
            // charge state data
            chargeStateDataBuffer.reset(new S_ChargeStateDataBuffer());
            chargeStateOrgaDataBuffer.reset(new S_ChargeStateOrgaDataBuffer());

            // atomic property data
            atomicStateDataBuffer.reset(new S_AtomicStateDataBuffer(m_numberAtomicStates));
            // atomic orga data
            atomicStateStartIndexBlockDataBuffer_BoundBound.reset(
                new S_AtomicStateStartIndexBlockDataBuffer_UpDown<ProcClassGroup::boundBoundBased>(
                    m_numberAtomicStates));
            atomicStateStartIndexBlockDataBuffer_BoundFree.reset(
                new S_AtomicStateStartIndexBlockDataBuffer_UpDown<ProcClassGroup::boundFreeBased>(
                    m_numberAtomicStates));
            atomicStateStartIndexBlockDataBuffer_Autonomous.reset(
                new S_AtomicStateStartIndexBlockDataBuffer_Down<ProcClassGroup::autonomousBased>(
                    m_numberAtomicStates));
            atomicStateNumberOfTransitionsDataBuffer_BoundBound.reset(
                new S_AtomicStateNumberOfTransitionsDataBuffer_UpDown<ProcClassGroup::boundBoundBased>(
                    m_numberAtomicStates));
            atomicStateNumberOfTransitionsDataBuffer_BoundFree.reset(
                new S_AtomicStateNumberOfTransitionsDataBuffer_UpDown<ProcClassGroup::boundFreeBased>(
                    m_numberAtomicStates));
            atomicStateNumberOfTransitionsDataBuffer_Autonomous.reset(
                new S_AtomicStateNumberOfTransitionsDataBuffer_Down<ProcClassGroup::autonomousBased>(
                    m_numberAtomicStates));

            // transition data
            boundBoundTransitionDataBuffer.reset(
                new S_BoundBoundTransitionDataBuffer<procClass::TransitionOrdering::byLowerState>(
                    m_numberBoundBoundTransitions));
            boundFreeTransitionDataBuffer.reset(
                new S_BoundFreeTransitionDataBuffer<procClass::TransitionOrdering::byLowerState>(
                    m_numberBoundFreeTransitions));
            autonomousTransitionDataBuffer.reset(
                new S_AutonomousTransitionDataBuffer<procClass::TransitionOrdering::byLowerState>(
                    m_numberAutonomousTransitions));

            inverseBoundBoundTransitionDataBuffer.reset(
                new S_BoundBoundTransitionDataBuffer<procClass::TransitionOrdering::byUpperState>(
                    m_numberBoundBoundTransitions));
            inverseBoundFreeTransitionDataBuffer.reset(
                new S_BoundFreeTransitionDataBuffer<procClass::TransitionOrdering::byUpperState>(
                    m_numberBoundFreeTransitions));
            inverseAutonomousTransitionDataBuffer.reset(
                new S_AutonomousTransitionDataBuffer<procClass::TransitionOrdering::byUpperState>(
                    m_numberAutonomousTransitions));

            // transition selection data
            transitionSelectionDataBuffer.reset(new S_TransitionSelectionDataBuffer(m_numberAtomicStates));
        }

        /** fill pure state property data storage buffer from list
         *
         * @tparam T_Tuple type of tuple
         * @tparam T_DataBox type of dataBox
         *
         * @param list correctly ordered list of data tuples to store
         * @attention does not sync to device, must be synced externally explicitly
         */
        template<typename T_Tuple, typename T_DataBox>
        ALPAKA_FN_HOST void storeStateData(std::list<T_Tuple>& list, T_DataBox hostBox)
        {
            typename std::list<T_Tuple>::iterator iter = list.begin();

            uint32_t i = 0u;

            for(; iter != list.end(); iter++)
            {
                hostBox.store(i, *iter);
                i++;
            }
        }

        /** fill pure transition property data storage buffer from list
         *
         * @tparam T_Tuple type of tuple
         * @tparam T_DataBox type of dataBox
         * @tparam T_StateDataBox type of atomicState property dataox
         *
         * @param list correctly ordered list of data tuples to store
         * @attention does not sync to device, must be synced externally explicitly
         */
        template<typename T_Tuple, typename T_DataBox, typename T_StateDataBox>
        ALPAKA_FN_HOST void storeTransitionData(
            std::list<T_Tuple>& list,
            T_DataBox hostBox,
            T_StateDataBox const stateHostBox)
        {
            typename std::list<T_Tuple>::iterator iter = list.begin();

            uint32_t i = 0u;

            for(; iter != list.end(); iter++)
            {
                hostBox.store(i, *iter, stateHostBox);
                i++;
            }
        }

        /** fill the charge orga data buffer
         *
         * @attention assumes that the atomic states are sorted block wise by charge state
         *
         * @param atomicStateList list of all atomicStates, sorted block wise by charge state
         */
        ALPAKA_FN_HOST void fillChargeStateOrgaData(std::list<S_AtomicStateTuple>& atomicStateList)
        {
            typename std::list<S_AtomicStateTuple>::iterator iter = atomicStateList.begin();

            S_ChargeStateOrgaDataBox chargeStateOrgaDataHostBox = chargeStateOrgaDataBuffer->getHostDataBox();

            uint8_t currentChargeState;

            // empty atomic state list
            if(iter == atomicStateList.end())
                return;

            // read first entry as first last entry
            uint8_t lastChargeState = ConfigNumber::getChargeState(std::get<0>(*iter));

            TypeNumber numberStates = 1u;
            TypeNumber startIndexLastBlock = 0u;
            iter++;

            // iterate over rest of the list
            TypeNumber i = 1u;
            for(; iter != atomicStateList.end(); iter++)
            {
                currentChargeState = ConfigNumber::getChargeState(std::get<0>(*iter));

                if(currentChargeState != lastChargeState)
                {
                    // new block
                    chargeStateOrgaDataHostBox.store(lastChargeState, numberStates, startIndexLastBlock);
                    numberStates = 1u;
                    startIndexLastBlock = i;
                    lastChargeState = currentChargeState;
                }
                else
                {
                    // same block
                    numberStates += 1u;
                }

                i++;
            }
            // finish last block
            chargeStateOrgaDataHostBox.store(lastChargeState, numberStates, startIndexLastBlock);

            chargeStateOrgaDataBuffer->hostToDevice();
        }

        /** fill the upward atomic state orga buffers for a transition groups
         *
         * i.e. number of transitions and start index, up, for each atomic state
         *  for a transition group(either bound-bound or bound-free)
         *
         * @tparam T_Tuple transition tuple type
         * @tparam T_NumberHostBox host data box of number of transitions buffer to fill transitions into
         * @tparam T_StartIndexHostBox host data box of start index block of transitions buffer to fill transitions
         * into
         *
         * @attention assumes that transitionList is sorted by lower state block wise
         * @attention changes have to synced to device separately
         * @attention startIndexBlock of a state is initialized to 0 if no transition in the
         *  up direction exist in the transition list
         *
         * @details implemented as a block accumulator iteration with two support points.
         *  The transition list is assumed to consist of strict-totally ordered(by lower state)
         *  blocks of transitions with each block of transitions sharing the same lower state.
         *
         *  The first support points stores the current open transition blocks lower state,
         *  while the second support point advances element-wise over the transition list
         *  until it finds a transition with lower state that is not equal to the current
         *  transition. This signifies the end of the open transition block.
         *  We then note down the accumulated values for the open block, close it, and
         *  open a new block, by setting the first support point to the second support point
         *  and continue as before until we reach the end of the transition list.
         */
        template<typename T_Tuple, ProcClassGroup T_ProcessClassGroup>
        ALPAKA_FN_HOST void fill_UpTransition_OrgaData(
            std::list<T_Tuple> transitionList,
            S_AtomicStateNumberOfTransitionsDataBox_UpDown<T_ProcessClassGroup> numberHostBox,
            S_AtomicStateStartIndexBlockDataBox_UpDown<T_ProcessClassGroup> startIndexHostBox)
        {
            typename std::list<T_Tuple>::iterator iter = transitionList.begin();

            // lookup data
            S_AtomicStateDataBox atomicStateDataHostBox = atomicStateDataBuffer->getHostDataBox();
            S_ChargeStateOrgaDataBox chargeStateOrgaDataHostBox = chargeStateOrgaDataBuffer->getHostDataBox();

            // prefill with zeros
            for(uint32_t j = 0u; j < m_numberAtomicStates; j++)
            {
                startIndexHostBox.storeUp(j, 0u);
                numberHostBox.storeUp(j, 0u);
            }

            uint8_t lastChargeState;
            uint32_t lastAtomicStateCollectionIndex;
            // transitions up from a state have the state as lower state
            Idx currentLower;

            // check for empty transition list
            if(iter == transitionList.end())
            {
                return;
            }

            // read first entry, always legal since
            Idx lastLower = getLowerStateConfigNumber<Idx, TypeValue>(*iter);
            TypeNumber numberInBlock = 1u;
            TypeNumber lastStartIndex = 0u;
            iter++;

            // init states before start of first block
            lastChargeState = ConfigNumber::getChargeState(lastLower);
            lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                lastLower,
                chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

            // iterate over rest of the list
            TypeNumber i = 1u;
            for(; iter != transitionList.end(); iter++)
            {
                currentLower = getLowerStateConfigNumber<Idx, TypeValue>(*iter);

                if(currentLower != lastLower)
                {
                    // new lower/upper state transition block
                    {
                        lastChargeState = ConfigNumber::getChargeState(lastLower);

                        lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                            lastLower,
                            chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                            chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

                        // check for state in transition but not defined in atomic state list
                        if(lastAtomicStateCollectionIndex >= atomicStateDataHostBox.numberAtomicStatesTotal())
                            throw std::runtime_error("atomicPhysics ERROR: atomic state not found");

                        startIndexHostBox.storeUp(lastAtomicStateCollectionIndex, lastStartIndex);
                        numberHostBox.storeUp(lastAtomicStateCollectionIndex, numberInBlock);

                        numberInBlock = 1u;
                        lastStartIndex = i;
                        lastLower = currentLower;
                    }
                }
                else
                {
                    // same lower/upper state transition block
                    numberInBlock += 1u;
                }

                i++;
            }

            // finish last block
            lastChargeState = ConfigNumber::getChargeState(lastLower);

            lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                lastLower,
                // completely ionized state can never be lower state of an transition
                chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

            startIndexHostBox.storeUp(lastAtomicStateCollectionIndex, lastStartIndex);
            numberHostBox.storeUp(lastAtomicStateCollectionIndex, numberInBlock);
            lastStartIndex = i;
        }

        /** fill the downward atomic state orga buffers for a transition groups
         *
         * i.e. number of transitions and start index, down, of each atomic state
         *  for a transition group(either bound-bound, bound-free or autonomous)
         *
         * @tparam T_Tuple transition tuple type
         * @tparam T_NumberHostBox host data box of number of transitions buffer to fill transitions into
         * @tparam T_StartIndexHostBox host data box of start index block of transitions buffer to fill transitions
         * into
         *
         * @attention assumes that transitionList is sorted by primarily upper state block wise and secondary by lower
         * state
         * @attention changes have to be synced to device separately
         * @attention startIndexBlock of a state is initialized to 0 if no transition in the
         *  down direction exist in the transition list
         *
         * @details see fill_UpTransition_OrgaData but instead of ordering blocks by lower
         *  state we order by upper state
         */
        template<
            typename T_Tuple,
            typename T_NumberHostBox,
            typename T_StartIndexHostBox,
            ProcClassGroup T_ProcessClassGroup>
        ALPAKA_FN_HOST void fill_DownTransition_OrgaData(
            std::list<T_Tuple> transitionList,
            T_NumberHostBox numberHostBox,
            T_StartIndexHostBox startIndexHostBox)
        {
            // check consistency of intended processClass group and given boxes
            PMACC_CASSERT_MSG(
                inconsistent_indicated_processClassGroup_and_passed_numberHostBox,
                u8(T_NumberHostBox::processClassGroup) == u8(T_ProcessClassGroup));
            PMACC_CASSERT_MSG(
                inconsistent_indicated_processClassGroup_and_passed_startIndexHostBox,
                (u8(T_StartIndexHostBox::processClassGroup) == u8(T_ProcessClassGroup)));

            typename std::list<T_Tuple>::iterator iter = transitionList.begin();

            S_AtomicStateDataBox atomicStateDataHostBox = atomicStateDataBuffer->getHostDataBox();
            S_ChargeStateOrgaDataBox chargeStateOrgaDataHostBox = chargeStateOrgaDataBuffer->getHostDataBox();

            // prefill with zeros
            for(uint32_t j = 0u; j < m_numberAtomicStates; j++)
            {
                startIndexHostBox.storeDown(j, 0u);
                numberHostBox.storeDown(j, 0u);
            }

            // empty transition list
            if(iter == transitionList.end())
            {
                return;
            }

            uint8_t lastChargeState;
            uint32_t lastAtomicStateCollectionIndex;
            Idx currentUpper;

            // read first entry
            Idx lastUpper = getUpperStateConfigNumber<Idx, TypeValue>(
                *iter); // transitions down from a state have the state as upper
            TypeNumber numberInBlock = 1u;
            TypeNumber lastStartIndex = 0u;
            iter++;

            // init states before start of first block
            lastChargeState = ConfigNumber::getChargeState(lastUpper);
            lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                lastUpper,
                chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

            // iterate over rest of the list
            TypeNumber i = 1u;
            for(; iter != transitionList.end(); iter++)
            {
                currentUpper = getUpperStateConfigNumber<Idx, TypeValue>(*iter);

                if(currentUpper != lastUpper)
                {
                    // new block
                    lastChargeState = ConfigNumber::getChargeState(lastUpper);

                    lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                        lastUpper,
                        chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                        chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

                    // check for state in transition but not defined in atomic state list
                    if(lastAtomicStateCollectionIndex >= atomicStateDataHostBox.numberAtomicStatesTotal())
                        throw std::runtime_error("atomicPhysics ERROR: atomic state not found");

                    startIndexHostBox.storeDown(lastAtomicStateCollectionIndex, lastStartIndex);
                    numberHostBox.storeDown(lastAtomicStateCollectionIndex, numberInBlock);

                    numberInBlock = 1u;
                    lastStartIndex = i;
                    lastUpper = currentUpper;
                }
                else
                {
                    // same block
                    numberInBlock += 1u;
                }

                i++;
            }

            // finish last block
            lastChargeState = ConfigNumber::getChargeState(lastUpper);

            lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                lastUpper,
                // completely ionized state can never be lower state of an transition
                chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

            startIndexHostBox.storeDown(lastAtomicStateCollectionIndex, lastStartIndex);
            numberHostBox.storeDown(lastAtomicStateCollectionIndex, numberInBlock);
            lastStartIndex = i;
        }

        /** fill the transition selectionBuffer
         *
         * @tparam electronicExcitation is channel active?
         * @tparam electronicDeexcitation is channel active?
         * @tparam spontaneousDeexcitation is channel active?
         * @tparam autonomousIonization is channel active?
         * @tparam electonicIonization is channel active?
         * @tparam fieldIonization is channel active?
         */
        template<
            bool electronicExcitation,
            bool electronicDeexcitation,
            bool spontaneousDeexcitation,
            bool electronicIonization,
            bool autonomousIonization,
            bool fieldIonization>
        ALPAKA_FN_HOST void fillTransitionSelectionDataBufferAndSetOffsets()
        {
            S_TransitionSelectionDataBox transitionSelectionDataHostBox
                = transitionSelectionDataBuffer->getHostDataBox();

            S_AtomicStateNumberOfTransitionsDataBox_UpDown<ProcClassGroup::boundBoundBased> hostBoxNumberBoundBound
                = atomicStateNumberOfTransitionsDataBuffer_BoundBound->getHostDataBox();
            S_AtomicStateNumberOfTransitionsDataBox_UpDown<ProcClassGroup::boundFreeBased> hostBoxNumberBoundFree
                = atomicStateNumberOfTransitionsDataBuffer_BoundFree->getHostDataBox();
            S_AtomicStateNumberOfTransitionsDataBox_Down<ProcClassGroup::autonomousBased> hostBoxNumberAutonomous
                = atomicStateNumberOfTransitionsDataBuffer_Autonomous->getHostDataBox();

            using S_NumberPhysicalTransitions
                = picongpu::particles::atomicPhysics2::processClass::NumberPhysicalTransitions<
                    electronicExcitation,
                    electronicDeexcitation,
                    spontaneousDeexcitation,
                    electronicIonization,
                    autonomousIonization,
                    fieldIonization>;

            TypeNumber numberPhysicalTransitionsTotal;

            for(uint32_t i = 0u; i < m_numberAtomicStates; i++)
            {
                // no-change transition
                numberPhysicalTransitionsTotal = static_cast<TypeNumber>(1u);

                // bound-bound transitions
                hostBoxNumberBoundBound.storeOffset(i, static_cast<TypeNumber>(1u));
                //      downward
                numberPhysicalTransitionsTotal += S_NumberPhysicalTransitions::getFactorBoundBoundDown()
                    * hostBoxNumberBoundBound.numberOfTransitionsDown(i);
                //      upward
                numberPhysicalTransitionsTotal += S_NumberPhysicalTransitions::getFactorBoundBoundUp()
                    * hostBoxNumberBoundBound.numberOfTransitionsUp(i);

                // bound-free transitions
                hostBoxNumberBoundFree.storeOffset(i, numberPhysicalTransitionsTotal);
                //      upward
                numberPhysicalTransitionsTotal += S_NumberPhysicalTransitions::getFactorBoundFreeUp()
                    * hostBoxNumberBoundFree.numberOfTransitionsUp(i);

                /// recombination missing, @todo implement recombination, Brian Marre

                // autonomousTransitions
                hostBoxNumberAutonomous.storeOffset(i, numberPhysicalTransitionsTotal);

                numberPhysicalTransitionsTotal += S_NumberPhysicalTransitions::getFactorAutonomous()
                    * hostBoxNumberAutonomous.numberOfTransitionsDown(i);

                transitionSelectionDataHostBox.store(i, numberPhysicalTransitionsTotal);
            }

            // sync offsets
            atomicStateNumberOfTransitionsDataBuffer_BoundBound->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_BoundFree->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_Autonomous->hostToDevice();

            // sync transition selection data
            transitionSelectionDataBuffer->hostToDevice();
        }

    public:
        /** read input files and create/fill data boxes
         *
         * @param fileChargeData path to file containing charge state data
         * @param fileAtomicStateData path to file containing atomic state data
         * @param fileBoundBoundTransitionData path to file containing bound-bound transition data
         * @param fileBoundFreeTransitionData path to file containing bound-free transition data
         * @param fileAutonomousTransitionData path to file containing autonomous transition data
         */
        AtomicData(
            std::string fileChargeStateData,
            std::string fileAtomicStateData,
            std::string fileBoundBoundTransitionData,
            std::string fileBoundFreeTransitionData,
            std::string fileAutonomousTransitionData,
            std::string speciesName)
            : m_speciesName(speciesName)
        {
            // read in files
            //      state data
            std::list<S_ChargeStateTuple> chargeStates = readChargeStates(fileChargeStateData);
            std::list<S_AtomicStateTuple> atomicStates = readAtomicStates(fileAtomicStateData);

            //      transition data
            std::list<S_BoundBoundTransitionTuple> boundBoundTransitions
                = readBoundBoundTransitions(fileBoundBoundTransitionData);
            std::list<S_BoundFreeTransitionTuple> boundFreeTransitions
                = readBoundFreeTransitions(fileBoundFreeTransitionData);
            std::list<S_AutonomousTransitionTuple> autonomousTransitions
                = readAutonomousTransitions(fileAutonomousTransitionData);

            //      sort by lower state of transition, optional since input files already sorted
            // boundBoundTransitions.sort(CompareTransitionTupel<TypeValue, Idx,true>());
            // boundFreeTransitions.sort(CompareTransitionTupel<TypeValue, Idx, true>());
            // autonomousTransitions.sort(CompareTransitionTupel<TypeValue, Idx,true>());

            // check assumptions
            checkChargeStateList(chargeStates);
            checkAtomicStateList(atomicStates);
            checkTransitionList<S_BoundBoundTransitionTuple>(boundBoundTransitions);
            checkTransitionList<S_BoundFreeTransitionTuple>(boundFreeTransitions);
            checkTransitionList<S_AutonomousTransitionTuple>(autonomousTransitions);

            // initialize buffers
            initBuffers();

            // fill data buffers
            //      states
            storeStateData<S_ChargeStateTuple, S_ChargeStateDataBox>(
                chargeStates,
                chargeStateDataBuffer->getHostDataBox());
            chargeStateDataBuffer->hostToDevice();

            storeStateData<S_AtomicStateTuple, S_AtomicStateDataBox>(
                atomicStates,
                atomicStateDataBuffer->getHostDataBox());
            atomicStateDataBuffer->hostToDevice();

            //      transitions
            storeTransitionData<
                S_BoundBoundTransitionTuple,
                S_BoundBoundTransitionDataBox<procClass::TransitionOrdering::byLowerState>,
                S_AtomicStateDataBox>(
                boundBoundTransitions,
                boundBoundTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            boundBoundTransitionDataBuffer->hostToDevice();

            storeTransitionData<
                S_BoundFreeTransitionTuple,
                S_BoundFreeTransitionDataBox<procClass::TransitionOrdering::byLowerState>,
                S_AtomicStateDataBox>(
                boundFreeTransitions,
                boundFreeTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            boundFreeTransitionDataBuffer->hostToDevice();

            storeTransitionData<
                S_AutonomousTransitionTuple,
                S_AutonomousTransitionDataBox<procClass::TransitionOrdering::byLowerState>,
                S_AtomicStateDataBox>(
                autonomousTransitions,
                autonomousTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            autonomousTransitionDataBuffer->hostToDevice();

            // check for inversions in upper lower state of transitions
            checkTransitionsForEnergyInversion(this->getBoundBoundTransitionDataBox<
                                               /*host*/ true,
                                               procClass::TransitionOrdering::byLowerState>());
            // no check for bound-free, since bound-free transitions may reduce overall energy
            checkTransitionsForEnergyInversion(this->getAutonomousTransitionDataBox<
                                               /*host*/ true,
                                               procClass::TransitionOrdering::byLowerState>());

            // fill orga data buffers 1,)
            //          charge state
            fillChargeStateOrgaData(atomicStates);

            //          atomic states, up direction
            fill_UpTransition_OrgaData<S_BoundBoundTransitionTuple, ProcClassGroup::boundBoundBased>(
                boundBoundTransitions,
                atomicStateNumberOfTransitionsDataBuffer_BoundBound->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_BoundBound->getHostDataBox());
            //              sync to device
            atomicStateNumberOfTransitionsDataBuffer_BoundBound->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_BoundBound->hostToDevice();

            fill_UpTransition_OrgaData<S_BoundFreeTransitionTuple, ProcClassGroup::boundFreeBased>(
                boundFreeTransitions,
                atomicStateNumberOfTransitionsDataBuffer_BoundFree->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_BoundFree->getHostDataBox());
            //              sync to device
            atomicStateNumberOfTransitionsDataBuffer_BoundFree->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_BoundFree->hostToDevice();

            // autonomous transitions are always only downward

            // re-sort by upper state of transition
            boundBoundTransitions.sort(CompareTransitionTupel<TypeValue, T_ConfigNumber, false>());
            boundFreeTransitions.sort(CompareTransitionTupel<TypeValue, T_ConfigNumber, false>());
            autonomousTransitions.sort(CompareTransitionTupel<TypeValue, T_ConfigNumber, false>());

            // store transition data in inverse order
            storeTransitionData<
                S_BoundBoundTransitionTuple,
                S_BoundBoundTransitionDataBox<procClass::TransitionOrdering::byUpperState>,
                S_AtomicStateDataBox>(
                boundBoundTransitions,
                inverseBoundBoundTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            inverseBoundBoundTransitionDataBuffer->hostToDevice();

            storeTransitionData<
                S_BoundFreeTransitionTuple,
                S_BoundFreeTransitionDataBox<procClass::TransitionOrdering::byUpperState>,
                S_AtomicStateDataBox>(
                boundFreeTransitions,
                inverseBoundFreeTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            inverseBoundFreeTransitionDataBuffer->hostToDevice();

            storeTransitionData<
                S_AutonomousTransitionTuple,
                S_AutonomousTransitionDataBox<procClass::TransitionOrdering::byUpperState>,
                S_AtomicStateDataBox>(
                autonomousTransitions,
                inverseAutonomousTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            inverseAutonomousTransitionDataBuffer->hostToDevice();

            // fill orga data buffers 2.)
            //      atomic states, down direction
            //      bound-bound
            fill_DownTransition_OrgaData<
                S_BoundBoundTransitionTuple,
                S_AtomicStateNumberOfTransitionsDataBox_UpDown<ProcClassGroup::boundBoundBased>,
                S_AtomicStateStartIndexBlockDataBox_UpDown<ProcClassGroup::boundBoundBased>,
                ProcClassGroup::boundBoundBased>(
                boundBoundTransitions,
                atomicStateNumberOfTransitionsDataBuffer_BoundBound->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_BoundBound->getHostDataBox());
            //          sync to device
            atomicStateNumberOfTransitionsDataBuffer_BoundBound->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_BoundBound->hostToDevice();

            //      bound-free
            fill_DownTransition_OrgaData<
                S_BoundFreeTransitionTuple,
                S_AtomicStateNumberOfTransitionsDataBox_UpDown<ProcClassGroup::boundFreeBased>,
                S_AtomicStateStartIndexBlockDataBox_UpDown<ProcClassGroup::boundFreeBased>,
                ProcClassGroup::boundFreeBased>(
                boundFreeTransitions,
                atomicStateNumberOfTransitionsDataBuffer_BoundFree->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_BoundFree->getHostDataBox());
            //          sync to device
            atomicStateNumberOfTransitionsDataBuffer_BoundFree->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_BoundFree->hostToDevice();

            //      autonomous
            fill_DownTransition_OrgaData<
                S_AutonomousTransitionTuple,
                S_AtomicStateNumberOfTransitionsDataBox_Down<ProcClassGroup::autonomousBased>,
                S_AtomicStateStartIndexBlockDataBox_Down<ProcClassGroup::autonomousBased>,
                ProcClassGroup::autonomousBased>(
                autonomousTransitions,
                atomicStateNumberOfTransitionsDataBuffer_Autonomous->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_Autonomous->getHostDataBox());
            //          sync to device
            atomicStateNumberOfTransitionsDataBuffer_Autonomous->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_Autonomous->hostToDevice();

            // fill transitionSelectionBuffer
            fillTransitionSelectionDataBufferAndSetOffsets<
                T_electronicExcitation,
                T_electronicDeexcitation,
                T_spontaneousDeexcitation,
                T_electronicIonization,
                T_autonomousIonization,
                T_fieldIonization>();

            if constexpr(picongpu::atomicPhysics2::debug::atomicData::DEBUG_SYNC_BUFFERS_TO_HOST)
                this->hostToDevice();
        }

        void hostToDevice()
        {
            // charge state data
            chargeStateDataBuffer->hostToDevice();
            chargeStateOrgaDataBuffer->hostToDevice();

            // atomic property data
            atomicStateDataBuffer->hostToDevice();
            // atomic orga data
            atomicStateStartIndexBlockDataBuffer_BoundBound->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_BoundFree->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_Autonomous->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_BoundBound->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_BoundFree->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_Autonomous->hostToDevice();

            // transition data
            boundBoundTransitionDataBuffer->hostToDevice();
            boundFreeTransitionDataBuffer->hostToDevice();
            autonomousTransitionDataBuffer->hostToDevice();

            // inverse transition data
            inverseBoundBoundTransitionDataBuffer->hostToDevice();
            inverseBoundFreeTransitionDataBuffer->hostToDevice();
            inverseAutonomousTransitionDataBuffer->hostToDevice();

            transitionSelectionDataBuffer->hostToDevice();
        }

        void deviceToHost()
        {
            // charge state data
            chargeStateDataBuffer->deviceToHost();
            chargeStateOrgaDataBuffer->deviceToHost();

            // atomic property data
            atomicStateDataBuffer->deviceToHost();
            // atomic orga data
            atomicStateStartIndexBlockDataBuffer_BoundBound->deviceToHost();
            atomicStateStartIndexBlockDataBuffer_BoundFree->deviceToHost();
            atomicStateStartIndexBlockDataBuffer_Autonomous->deviceToHost();
            atomicStateNumberOfTransitionsDataBuffer_BoundBound->deviceToHost();
            atomicStateNumberOfTransitionsDataBuffer_BoundFree->deviceToHost();
            atomicStateNumberOfTransitionsDataBuffer_Autonomous->deviceToHost();

            // transition data
            boundBoundTransitionDataBuffer->deviceToHost();
            boundFreeTransitionDataBuffer->deviceToHost();
            autonomousTransitionDataBuffer->deviceToHost();

            // inverse transition data
            inverseBoundBoundTransitionDataBuffer->deviceToHost();
            inverseBoundFreeTransitionDataBuffer->deviceToHost();
            inverseAutonomousTransitionDataBuffer->deviceToHost();

            transitionSelectionDataBuffer->deviceToHost();
        }

        // charge states
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_ChargeStateDataBox getChargeStateDataDataBox()
        {
            if constexpr(hostData)
                return chargeStateDataBuffer->getHostDataBox();
            else
                return chargeStateDataBuffer->getDeviceDataBox();

            ALPAKA_UNREACHABLE(chargeStateDataBuffer->getHostDataBox());
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_ChargeStateOrgaDataBox getChargeStateOrgaDataBox()
        {
            if constexpr(hostData)
                return chargeStateOrgaDataBuffer->getHostDataBox();
            else
                return chargeStateOrgaDataBuffer->getDeviceDataBox();

            ALPAKA_UNREACHABLE(chargeStateOrgaDataBuffer->getHostDataBox());
        }

        // atomic states
        //      property data
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateDataBox getAtomicStateDataDataBox()
        {
            if constexpr(hostData)
                return atomicStateDataBuffer->getHostDataBox();
            else
                return atomicStateDataBuffer->getDeviceDataBox();

            ALPAKA_UNREACHABLE(atomicStateDataBuffer->getHostDataBox());
        }

        //      start index orga data
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateStartIndexBlockDataBox_UpDown<ProcClassGroup::boundBoundBased>
        getBoundBoundStartIndexBlockDataBox()
        {
            if constexpr(hostData)
                return atomicStateStartIndexBlockDataBuffer_BoundBound->getHostDataBox();
            else
                return atomicStateStartIndexBlockDataBuffer_BoundBound->getDeviceDataBox();

            ALPAKA_UNREACHABLE(atomicStateStartIndexBlockDataBuffer_BoundBound->getHostDataBox());
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateStartIndexBlockDataBox_UpDown<ProcClassGroup::boundFreeBased> getBoundFreeStartIndexBlockDataBox()
        {
            if constexpr(hostData)
                return atomicStateStartIndexBlockDataBuffer_BoundFree->getHostDataBox();
            else
                return atomicStateStartIndexBlockDataBuffer_BoundFree->getDeviceDataBox();

            ALPAKA_UNREACHABLE(atomicStateStartIndexBlockDataBuffer_BoundFree->getHostDataBox());
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateStartIndexBlockDataBox_Down<ProcClassGroup::autonomousBased> getAutonomousStartIndexBlockDataBox()
        {
            if constexpr(hostData)
                return atomicStateStartIndexBlockDataBuffer_Autonomous->getHostDataBox();
            else
                return atomicStateStartIndexBlockDataBuffer_Autonomous->getDeviceDataBox();

            ALPAKA_UNREACHABLE(atomicStateStartIndexBlockDataBuffer_Autonomous->getHostDataBox());
        }

        //      number transitions orga data
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateNumberOfTransitionsDataBox_UpDown<ProcClassGroup::boundBoundBased>
        getBoundBoundNumberTransitionsDataBox()
        {
            if constexpr(hostData)
                return atomicStateNumberOfTransitionsDataBuffer_BoundBound->getHostDataBox();
            else
                return atomicStateNumberOfTransitionsDataBuffer_BoundBound->getDeviceDataBox();

            ALPAKA_UNREACHABLE(atomicStateNumberOfTransitionsDataBuffer_BoundBound->getHostDataBox());
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateNumberOfTransitionsDataBox_UpDown<ProcClassGroup::boundFreeBased>
        getBoundFreeNumberTransitionsDataBox()
        {
            if constexpr(hostData)
                return atomicStateNumberOfTransitionsDataBuffer_BoundFree->getHostDataBox();
            else
                return atomicStateNumberOfTransitionsDataBuffer_BoundFree->getDeviceDataBox();

            ALPAKA_UNREACHABLE(atomicStateNumberOfTransitionsDataBuffer_BoundFree->getHostDataBox());
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateNumberOfTransitionsDataBox_Down<ProcClassGroup::autonomousBased>
        getAutonomousNumberTransitionsDataBox()
        {
            if constexpr(hostData)
                return atomicStateNumberOfTransitionsDataBuffer_Autonomous->getHostDataBox();
            else
                return atomicStateNumberOfTransitionsDataBuffer_Autonomous->getDeviceDataBox();

            ALPAKA_UNREACHABLE(atomicStateNumberOfTransitionsDataBuffer_Autonomous->getHostDataBox());
        }

        // transition data
        /** bound-bound transition data access
         *
         * @tparam hostData true =^= get hostDataBox, false =^= get DeviceDataBox
         * @tparam T_TransitionOrdering, get data box ordered by lower or upper state,
         *  ordering defined by CompareTransitionTupel implementation
         */
        template<bool hostData, procClass::TransitionOrdering T_TransitionOrdering>
        S_BoundBoundTransitionDataBox<T_TransitionOrdering> getBoundBoundTransitionDataBox()
        {
            constexpr bool byLowerState
                = (u8(T_TransitionOrdering) == u8(procClass::TransitionOrdering::byLowerState));

            if constexpr(byLowerState)
            {
                if constexpr(hostData)
                    return boundBoundTransitionDataBuffer->getHostDataBox();
                else
                    return boundBoundTransitionDataBuffer->getDeviceDataBox();
            }
            else
            {
                if constexpr(hostData)
                    return inverseBoundBoundTransitionDataBuffer->getHostDataBox();
                else
                    return inverseBoundBoundTransitionDataBuffer->getDeviceDataBox();
            }
            ALPAKA_UNREACHABLE(S_BoundBoundTransitionDataBuffer<T_TransitionOrdering>(0u).getHostDataBox());
        }

        /** bound-free transition data access
         *
         * @tparam hostData true =^= get hostDataBox, false =^= get DeviceDataBox
         * @tparam T_TransitionOrdering, get data box ordered by lower or upper state,
         *  ordering defined by CompareTransitionTupel implementation
         */
        template<bool hostData, procClass::TransitionOrdering T_TransitionOrdering>
        S_BoundFreeTransitionDataBox<T_TransitionOrdering> getBoundFreeTransitionDataBox()
        {
            constexpr bool byLowerState
                = (u8(T_TransitionOrdering) == u8(procClass::TransitionOrdering::byLowerState));

            if constexpr(byLowerState)
            {
                if constexpr(hostData)
                    return boundFreeTransitionDataBuffer->getHostDataBox();
                else
                    return boundFreeTransitionDataBuffer->getDeviceDataBox();
            }
            else
            {
                if constexpr(hostData)
                    return inverseBoundFreeTransitionDataBuffer->getHostDataBox();
                else
                    return inverseBoundFreeTransitionDataBuffer->getDeviceDataBox();
            }
            ALPAKA_UNREACHABLE(S_BoundFreeTransitionDataBuffer<T_TransitionOrdering>(0u).getHostDataBox());
        }

        /** autonomous transition data access
         *
         * @tparam hostData true =^= get hostDataBox, false =^= get DeviceDataBox
         * @tparam T_TransitionOrdering, get data box ordered by lower or upper state,
         *  ordering defined by CompareTransitionTupel implementation
         */
        template<bool hostData, procClass::TransitionOrdering T_TransitionOrdering>
        S_AutonomousTransitionDataBox<T_TransitionOrdering> getAutonomousTransitionDataBox()
        {
            constexpr bool byLowerState
                = (u8(T_TransitionOrdering) == u8(procClass::TransitionOrdering::byLowerState));

            if constexpr(byLowerState)
            {
                if constexpr(hostData)
                    return autonomousTransitionDataBuffer->getHostDataBox();
                else
                    return autonomousTransitionDataBuffer->getDeviceDataBox();
            }
            else
            {
                if constexpr(hostData)
                    return inverseAutonomousTransitionDataBuffer->getHostDataBox();
                else
                    return inverseAutonomousTransitionDataBuffer->getDeviceDataBox();
            }
            ALPAKA_UNREACHABLE(S_AutonomousTransitionDataBuffer<T_TransitionOrdering>(0u).getHostDataBox());
        }

        // transition selection data
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_TransitionSelectionDataBox getTransitionSelectionDataBox()
        {
            if constexpr(hostData)
                return transitionSelectionDataBuffer->getHostDataBox();
            else
                return transitionSelectionDataBuffer->getDeviceDataBox();

            ALPAKA_UNREACHABLE(transitionSelectionDataBuffer->getHostDataBox());
        }

        // debug queries
        uint32_t getNumberAtomicStates() const
        {
            return m_numberAtomicStates;
        }

        uint32_t getNumberBoundBoundTransitions() const
        {
            return m_numberBoundBoundTransitions;
        }

        uint32_t getNumberBoundFreeTransitions() const
        {
            return m_numberBoundFreeTransitions;
        }

        uint32_t getNumberAutonomousTransitions() const
        {
            return m_numberAutonomousTransitions;
        }

        //! == deviceToHost, required by ISimulationData
        void synchronize() override
        {
            this->deviceToHost();
        }

        //! required by ISimulationData
        std::string getUniqueId() override
        {
            return m_speciesName + "_atomicData";
        }
    };

} // namespace picongpu::particles::atomicPhysics2::atomicData
