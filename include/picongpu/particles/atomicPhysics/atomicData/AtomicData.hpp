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

#include "picongpu/defines.hpp" // need: picongpu/param/atomicPhysics_Debug.param

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/static_assert.hpp>

// charge state data
#include "picongpu/particles/atomicPhysics/atomicData/ChargeStateData.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/ChargeStateOrgaData.hpp"

// atomic state data
#include "picongpu/particles/atomicPhysics/atomicData/AtomicStateData.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicStateNumberOfTransitionsData_Down.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicStateNumberOfTransitionsData_UpDown.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicStateStartIndexBlockData_Down.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicStateStartIndexBlockData_UpDown.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/IPDIonizationStateData.hpp"

// transition data
#include "picongpu/particles/atomicPhysics/atomicData/AutonomousTransitionData.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/BoundBoundTransitionData.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/BoundFreeTransitionData.hpp"

// tuple definitions
#include "picongpu/particles/atomicPhysics/atomicData/AtomicTuples.def"
// helper stuff for transition tuples
#include "picongpu/particles/atomicPhysics/atomicData/CheckTransitionTuple.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/CompareTransitionTuple.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/GetStateFromTransitionTuple.hpp"

// enums for configuration and meta description
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics/enums/ADKLaserPolarization.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClassGroup.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"

// debug only
#include "picongpu/particles/atomicPhysics/debug/PrintTransitionTupleToConsole.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

//! @file gathers atomic data storage implementations and implements filling them on runtime

namespace picongpu::particles::atomicPhysics::atomicData
{
    namespace detail
    {
        enum struct StorageDirectionSwitch : uint8_t
        {
            none,
            upward,
            downward,
        };
    } // namespace detail

    namespace s_enums = picongpu::particles::atomicPhysics::enums;
    using ProcClassGroup = picongpu::particles::atomicPhysics::enums::ProcessClassGroup;

    /** gathering of all atomicPhysics input data
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_CollectionIndex dataType used for collection index, typically uint32_t
     * @tparam T_ConfigNumber type holding definition of atomicConfigNumber for species
     *  see picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber.hpp
     * @tparam T_Multiplicities dataType used for storage of stae multiplicities, typically float64
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
     * These model's parameters are provided by the user in .txt files of specified format
     * (see atomicPhysics model documentation) at runtime.
     *
     *  PIConGPU itself only includes charge state data, for ADK-, Thomas-Fermi- and BSI-ionization.
     *  All other atomic state data is kept separate from PIConGPU itself, due to licensing requirements.
     *
     * These files are read at the start of the simulation and stored distributed over the following:
     *  - charge state property data [ChargeStateData.hpp]
     *      * ionization energy
     *      * screened charge
     *  - charge state orga data [ChargeStateOrgaData.hpp]
     *      * number of atomic states for each charge state
     *      * start index block for charge state in list of atomic states
     * - atomic state property data [AtomicStateData.hpp]
     *      * configNumber
     *      * state energy, above ground state of charge state
     * - atomic state orga data
     *      [AtomicStateNumberOfTransitionsData_Down.hpp, AtomicStateNumberOfTransitionsData_UpDown.hpp]
     *       * number of transitions (up-/)down for each atomic state,
     *          by type of transition(bound-bound/bound-free/autonomous)
     *      [AtomicStateStartIndexBlockData_Down.hpp, AtomicStateStartIndexBlockData_UpDown.hpp]
     *       * start index of atomic state's block of transitions in transition collection,
     *          by type of transition(bound-bound/bound-free/autonomous)
     * - pressure ionization data [IPDIonizationData.hpp]
     *      * pressure ionization state collectionIndex
     * - transition property data[BoundBoundTransitionData.hpp, BoundFreeTransitionData.hpp,
     *      AutonomousTransitionData.hpp]
     *      * parameters for cross section calculation for each modelled transition
     *
     * @note orga data describes the structure of the property data for faster lookups, lookups are always possible
     *       without it, but are possibly non performant
     *
     * For each of data subsets exists a dataBox class, a container class holding pmacc::dataBox'es, and a dataBuffer
     *  class, a container class holding pmacc::buffers in turn allowing access to the pmacc::dataBox'es.
     *
     * Each dataBuffer will create on demand a host- or device-side dataBox objects which in turn gives access to the
     *  data.
     *
     * Assumptions about input data are described in CheckTransitionTuple.hpp, ordering requirements of transitions in
     *  CompareTransitionTuple.hpp and all other requirements in the checkChargeStateList(), checkAtomicStateList() and
     *  checkTransitionsForEnergyInversion() methods.
     *
     * @todo add photonic channels, Brian Marre, 2022
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
        bool T_fieldIonization,
        atomicPhysics::enums::ADKLaserPolarization T_ADKLaserPolarization>
    class AtomicData : public pmacc::ISimulationData
    {
    public:
        using TypeNumber = T_Number;
        using TypeValue = T_Value;
        using CollectionIdx = T_CollectionIndex;
        using Idx = typename T_ConfigNumber::DataType;
        using ConfigNumber = T_ConfigNumber;
        using Multiplicities = T_Multiplicities;

        static constexpr bool switchElectronicExcitation = T_electronicExcitation;
        static constexpr bool switchElectronicDeexcitation = T_electronicDeexcitation;
        static constexpr bool switchSpontaneousDeexcitation = T_spontaneousDeexcitation;
        static constexpr bool switchElectronicIonization = T_electronicIonization;
        static constexpr bool switchAutonomousIonization = T_autonomousIonization;
        static constexpr bool switchFieldIonization = T_fieldIonization;
        static constexpr s_enums::ADKLaserPolarization ADKLaserPolarization = T_ADKLaserPolarization;

        /// type shorthand definitions
        //@{
        // tuples: S_* for shortened name
        using S_ChargeStateTuple = ChargeStateTuple<TypeValue>;
        using S_AtomicStateTuple = AtomicStateTuple<TypeValue, Idx>;
        using S_IPDIonizationStateTuple = IPDIonizationStateTuple<Idx>;
        using S_BoundBoundTransitionTuple = BoundBoundTransitionTuple<TypeValue, Idx>;
        using S_BoundFreeTransitionTuple = BoundFreeTransitionTuple<TypeValue, Idx>;
        using S_AutonomousTransitionTuple = AutonomousTransitionTuple<Idx>;

        // dataBuffers: S_* for shortened name
        using S_ChargeStateDataBuffer = ChargeStateDataBuffer<TypeNumber, TypeValue, ConfigNumber::atomicNumber>;
        using S_ChargeStateOrgaDataBuffer
            = ChargeStateOrgaDataBuffer<TypeNumber, TypeValue, T_ConfigNumber::atomicNumber>;
        using S_AtomicStateDataBuffer = AtomicStateDataBuffer<TypeNumber, TypeValue, ConfigNumber, Multiplicities>;

        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateStartIndexBlockDataBuffer_UpDown
            = AtomicStateStartIndexBlockDataBuffer_UpDown<CollectionIdx, TypeValue, T_ProcessClassGroup>;
        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateStartIndexBlockDataBuffer_Down
            = AtomicStateStartIndexBlockDataBuffer_Down<CollectionIdx, TypeValue, T_ProcessClassGroup>;
        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateNumberOfTransitionsDataBuffer_UpDown
            = AtomicStateNumberOfTransitionsDataBuffer_UpDown<TypeNumber, TypeValue, T_ProcessClassGroup>;
        template<ProcClassGroup T_ProcessClassGroup>
        using S_AtomicStateNumberOfTransitionsDataBuffer_Down
            = AtomicStateNumberOfTransitionsDataBuffer_Down<TypeNumber, TypeValue, T_ProcessClassGroup>;

        using S_IPDIonizationStateDataBuffer = IPDIonizationStateDataBuffer<CollectionIdx>;

        template<s_enums::TransitionOrdering T_TransitionOrdering>
        using S_BoundBoundTransitionDataBuffer = BoundBoundTransitionDataBuffer<
            TypeNumber,
            TypeValue,
            CollectionIdx,
            typename ConfigNumber::DataType,
            T_TransitionOrdering>;
        template<s_enums::TransitionOrdering T_TransitionOrdering>
        using S_BoundFreeTransitionDataBuffer = BoundFreeTransitionDataBuffer<
            TypeNumber,
            TypeValue,
            CollectionIdx,
            ConfigNumber,
            Multiplicities,
            T_TransitionOrdering>;
        template<s_enums::TransitionOrdering T_TransitionOrdering>
        using S_AutonomousTransitionDataBuffer = AutonomousTransitionDataBuffer<
            TypeNumber,
            TypeValue,
            T_CollectionIndex,
            typename T_ConfigNumber::DataType,
            T_TransitionOrdering>;

        // dataBoxes: S_* for shortened name
        using S_ChargeStateDataBox = ChargeStateDataBox<TypeNumber, TypeValue, ConfigNumber::atomicNumber>;
        using S_ChargeStateOrgaDataBox = ChargeStateOrgaDataBox<TypeNumber, TypeValue, ConfigNumber::atomicNumber>;

        using S_AtomicStateDataBox = AtomicStateDataBox<TypeNumber, TypeValue, ConfigNumber, Multiplicities>;

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

        using S_IPDIonizationStateDataBox = IPDIonizationStateDataBox<CollectionIdx>;

        template<s_enums::TransitionOrdering T_TransitionOrdering>
        using S_BoundBoundTransitionDataBox = BoundBoundTransitionDataBox<
            TypeNumber,
            TypeValue,
            CollectionIdx,
            typename ConfigNumber::DataType,
            T_TransitionOrdering>;
        template<s_enums::TransitionOrdering T_TransitionOrdering>
        using S_BoundFreeTransitionDataBox = BoundFreeTransitionDataBox<
            TypeNumber,
            TypeValue,
            CollectionIdx,
            ConfigNumber,
            Multiplicities,
            T_TransitionOrdering>;
        template<s_enums::TransitionOrdering T_TransitionOrdering>
        using S_AutonomousTransitionDataBox = AutonomousTransitionDataBox<
            TypeNumber,
            TypeValue,
            CollectionIdx,
            typename ConfigNumber::DataType,
            T_TransitionOrdering>;
        //@}
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

        // pressure ionization states
        std::unique_ptr<S_IPDIonizationStateDataBuffer> ipdIonizationStateDataBuffer;

        // transition data, normal, sorted by lower state
        std::unique_ptr<S_BoundBoundTransitionDataBuffer<s_enums::TransitionOrdering::byLowerState>>
            boundBoundTransitionDataBuffer;
        std::unique_ptr<S_BoundFreeTransitionDataBuffer<s_enums::TransitionOrdering::byLowerState>>
            boundFreeTransitionDataBuffer;
        std::unique_ptr<S_AutonomousTransitionDataBuffer<s_enums::TransitionOrdering::byLowerState>>
            autonomousTransitionDataBuffer;

        // transition data, inverted,sorted by upper state
        std::unique_ptr<S_BoundBoundTransitionDataBuffer<s_enums::TransitionOrdering::byUpperState>>
            inverseBoundBoundTransitionDataBuffer;
        std::unique_ptr<S_BoundFreeTransitionDataBuffer<s_enums::TransitionOrdering::byUpperState>>
            inverseBoundFreeTransitionDataBuffer;
        std::unique_ptr<S_AutonomousTransitionDataBuffer<s_enums::TransitionOrdering::byUpperState>>
            inverseAutonomousTransitionDataBuffer;

        uint32_t m_numberAtomicStates = 0u;

        uint32_t m_numberBoundBoundTransitions = 0u;
        uint32_t m_numberBoundFreeTransitions = 0u;
        uint32_t m_numberAutonomousTransitions = 0u;

        const std::string m_speciesName;

        //! try to open file, otherwise throw error
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
         *   - charge state data is sorted ascending by charge
         *   - the completely ionized state is left out
         *
         * @throws runtime error if file not found/accessible
         * @returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_ChargeStateTuple> readChargeStates(std::string const fileName)
        {
            std::ifstream file = openFile(fileName, "charge state data");
            if(!file)
                return std::list<S_ChargeStateTuple>{};

            std::list<S_ChargeStateTuple> chargeStateList{};

            TypeValue ionizationEnergy;
            TypeValue screenedCharge;
            uint32_t chargeState;
            uint8_t numberChargeStates = 0u;

            while(file >> chargeState >> ionizationEnergy >> screenedCharge)
            {
                if(chargeState == u32(T_ConfigNumber::atomicNumber))
                    throw std::runtime_error(
                        "charge state " + std::to_string(chargeState)
                        + " should not be included in input file for Z = "
                        + std::to_string(T_ConfigNumber::atomicNumber));

                S_ChargeStateTuple item = std::make_tuple(
                    u8(chargeState),
                    ionizationEnergy, // [eV]
                    screenedCharge); // [e]

                chargeStateList.push_back(item);

                ++numberChargeStates;
            }

            if(numberChargeStates != T_ConfigNumber::atomicNumber)
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
         *
         * @throws runtime error if file not found/accessible
         * @returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_AtomicStateTuple> readAtomicStates(std::string const fileName)
        {
            std::ifstream file = openFile(fileName, "atomic state data");
            if(!file)
                return std::list<S_AtomicStateTuple>{};

            std::list<S_AtomicStateTuple> atomicStateList{};

            uint64_t stateConfigNumber;
            TypeValue energyOverGround;

            while(file >> stateConfigNumber >> energyOverGround)
            {
                S_AtomicStateTuple item = std::make_tuple(
                    static_cast<Idx>(stateConfigNumber), // unitless
                    energyOverGround); // [eV]

                atomicStateList.push_back(item);

                ++m_numberAtomicStates;
            }

            return atomicStateList;
        }

        /** read pressure ionization data file
         *
         * @attention assumes pressure ionization state list to be ordered ascending by collectionIndex of atomic state
         *
         * @throws runtime error if file not found/accessible
         * @returns empty list if fileName is empty string or file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_IPDIonizationStateTuple> readIPDIonizationStates(std::string const fileName)
        {
            std::ifstream file = openFile(fileName, "pressure ionization states");

            if(!file)
                return std::list<S_IPDIonizationStateTuple>{};

            std::list<S_IPDIonizationStateTuple> ipdIonizationStateList{};

            uint64_t stateConfigNumber;
            uint64_t ipdIonizationStateConfigNumber;

            while(file >> stateConfigNumber >> ipdIonizationStateConfigNumber)
            {
                S_IPDIonizationStateTuple item = std::make_tuple(
                    static_cast<Idx>(stateConfigNumber), // unitless
                    static_cast<Idx>(ipdIonizationStateConfigNumber)); // unitless

                ipdIonizationStateList.push_back(item);
            }
            return ipdIonizationStateList;
        }

        /** read bound-bound transitions data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - transition data is sorted block wise ascending by lower atomic state and secondary ascending by upper
         * state configNumber
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
                ++m_numberBoundBoundTransitions;
            }

            return boundBoundTransitions;
        }

        /** read bound-free transitions data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - transition data is sorted block wise ascending by lower atomic state and secondary ascending by upper
         * state configNumber
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
         *   - transition data is sorted block wise ascending by lower atomic state and secondary ascending by upper
         * state configNumber
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

        /** check pressure ionization state list
         *
         * @return passes silently if correct
         *
         * @attention assumes previous fill of atomic state data buffer and charge state orga data buffer
         *
         * @throws runtime error if list not ordered by atomic state collectionIndex, pressure ionization state
         *  missing from atomic state input, or atomic state from atomic state input is missing in list.
         */
        ALPAKA_FN_HOST void checkIPDIonizationList(std::list<S_IPDIonizationStateTuple>& ipdIonizationStateList)
        {
            // check correct number of entries
            if(ipdIonizationStateList.size() != m_numberAtomicStates)
                throw std::runtime_error("atomicPhysics ERROR: number of pressure ionization states does not match "
                                         "number of atomic states");

            typename std::list<S_IPDIonizationStateTuple>::iterator iter = ipdIonizationStateList.begin();

            S_AtomicStateDataBox atomicStateDataHostBox = this->getAtomicStateDataDataBox<true>();
            S_ChargeStateOrgaDataBox chargeStateOrgaDataHostBox = this->getChargeStateOrgaDataBox<true>();


            for(CollectionIdx collectionIndexAtomicState = 0u; collectionIndexAtomicState < m_numberAtomicStates;
                ++collectionIndexAtomicState)
            {
                //! check order
                //@{
                //      always valid since, number elements in list == m_numberAtomicStates
                S_IPDIonizationStateTuple tupleIPDIonization = *iter;

                Idx stateConfigNumber = static_cast<Idx>(std::get<0>(tupleIPDIonization));

                if(stateConfigNumber != atomicStateDataHostBox.configNumber(collectionIndexAtomicState))
                {
                    std::string errorMessage = "atomicPhysics ERROR: mismatch between atomic state columns of atomic "
                                               "state input and pressure ionization input in element "
                        + std::to_string(collectionIndexAtomicState);
                    throw std::runtime_error(errorMessage);
                }
                //@}

                //! check pressure ionization state exists
                //! @note PI ... ipdIonizationState
                //@{
                //      get pressure ionization state collection index
                Idx const directPIconfigNumber = static_cast<Idx>(std::get<1>(tupleIPDIonization));
                uint8_t const PIchargeState = ConfigNumber::getChargeState(directPIconfigNumber);

                CollectionIdx collectionIndexDirectPIState = atomicStateDataHostBox.findStateCollectionIndex(
                    directPIconfigNumber,
                    chargeStateOrgaDataHostBox.numberAtomicStates(PIchargeState),
                    chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(PIchargeState));

                bool const configNumberNotFound = (collectionIndexDirectPIState == m_numberAtomicStates);
                if(configNumberNotFound)
                    throw std::runtime_error(
                        "atomicPhysics ERROR: pressure ionization state of state "
                        + std::to_string(atomicStateDataHostBox.configNumber(collectionIndexAtomicState))
                        + " not found in input atomic state data");
                //@}

                iter++;
            }
        }

        /** check transition list
         *
         * @param transitionList
         *
         * @attention assumes that chargeStateList and atomicStateList fulfill all ordering assumptions
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
                    picongpu::particles::atomicPhysics::debug::
                        printTransitionTupleToConsole<T_TransitionTuple, Idx, TypeValue, ConfigNumber>(
                            lastTransitionTuple);
                    picongpu::particles::atomicPhysics::debug::
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
                ((u8(T_TransitionHostBox::processClassGroup) == u8(s_enums::ProcessClassGroup::boundBoundBased))
                 || (u8(T_TransitionHostBox::processClassGroup) == u8(s_enums::ProcessClassGroup::autonomousBased))));

            uint32_t const numberTransitions = transitionHostBox.getNumberOfTransitionsTotal();

            for(uint32_t collectionIndex = u32(0u); collectionIndex < numberTransitions; collectionIndex++)
            {
                float_X const deltaEnergy = picongpu::particles::atomicPhysics::DeltaEnergyTransition::get(
                    collectionIndex,
                    this->getAtomicStateDataDataBox<true>(),
                    transitionHostBox,
                    // ionization potential depression
                    0._X,
                    this->getChargeStateDataDataBox<true>());

                if(deltaEnergy < 0._X)
                    throw std::runtime_error(
                        "atomicPhysics ERROR: upper and lower state inverted in energy in input"
                        ", energy lower state must be <= energy upper state, transitionType: "
                        + enumToString<T_TransitionHostBox::processClassGroup>() + " ,transition #"
                        + std::to_string(collectionIndex));
            }
        }

        /** fill ipdIonizationData buffer from existing list
         *
         * Takes a list of pressure ionization states for each atomic state in the input, looks up the collection index
         * of the pressure ionization states and stores for each atomic states, by collectionIndex, it's pressure
         * ionization state's collectionIndex.
         *
         * @param ipdIonizationStateList list of tuples(atomicConfigNumber of atomic state,
         *  atomicConfigNumber ipdIonization state)
         *
         * @attention assumes previous fill of atomic state data buffer and charge state orga data buffer
         * @attention assumes ipdIonizationStateList to be ordered ascending by collectionIndex of the atomic
         * state
         * @attention assumes that every atomic state in the atomic state input has exactly one associated pressure
         *  ionization state that exists in the atomic input data set
         */
        ALPAKA_FN_HOST void fillIPDIonizationDataFromList(std::list<S_IPDIonizationStateTuple>& ipdIonizationStateList)
        {
            typename std::list<S_IPDIonizationStateTuple>::iterator iter = ipdIonizationStateList.begin();

            S_AtomicStateDataBox atomicStateDataHostBox = this->getAtomicStateDataDataBox<true>();
            S_ChargeStateOrgaDataBox chargeStateOrgaDataHostBox = this->getChargeStateOrgaDataBox<true>();
            S_IPDIonizationStateDataBox ipdIonizationHostDataBox = this->getIPDIonizationStateDataBox<true>();

            CollectionIdx collectionIndexAtomicState = 0u;
            for(; iter != ipdIonizationStateList.end(); iter++)
            {
                S_IPDIonizationStateTuple tupleIPDIonization = *iter;

                // get pressure ionization state collection index
                Idx const ipdIonizationStateConfigNumber = static_cast<Idx>(std::get<1>(tupleIPDIonization));
                uint8_t const ipdIonizationStateChargeState
                    = ConfigNumber::getChargeState(ipdIonizationStateConfigNumber);

                // check is actually ionization, or disabled
                Idx const atomicStateConfigNumber = atomicStateDataHostBox.configNumber(collectionIndexAtomicState);
                uint8_t const atomicStateChargeState = ConfigNumber::getChargeState(atomicStateConfigNumber);

                bool const ipdIonizationIsDisabled = (ipdIonizationStateConfigNumber == atomicStateConfigNumber);
                bool const transitionIsIonization = ((atomicStateChargeState + 1) <= ipdIonizationStateChargeState);
                if(!transitionIsIonization && !ipdIonizationIsDisabled)
                    throw std::runtime_error(
                        "atomicPhysics ERROR: pressure ionization state[" + std::to_string(collectionIndexAtomicState)
                        + "] is no ionization state of corresponding atomic state");

                CollectionIdx collectionIndexIPDIonizationState = atomicStateDataHostBox.findStateCollectionIndex(
                    ipdIonizationStateConfigNumber,
                    chargeStateOrgaDataHostBox.numberAtomicStates(ipdIonizationStateChargeState),
                    chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(ipdIonizationStateChargeState));

                ipdIonizationHostDataBox.store(collectionIndexAtomicState, collectionIndexIPDIonizationState);

                ++collectionIndexAtomicState;
            }
        }

        /** fill ipdIonizationData buffer by automatically determining pressure ionization state
         *
         * chooses for each atomic state a pressure ionization state as, in descending priority:
         * 1.) direct pressure ionization state, if present in atomic state input
         * 2.) upper state of an upward bound-free transition starting from the state closest in energy to the lower
         *  state, if at least one boundfree transition exists
         * 3.) disable pressure ionization for the state
         *
         * @attention assumes previous fill of atomic state data buffer, charge state orga data buffer, the bound-free
         *  number transitions buffer, the bound-free startIndexBlock buffer and the bound-free transition buffer
         *
         * @note PI ... Pressure Ionization
         */
        ALPAKA_FN_HOST void fillIPDIonizationDataAuto()
        {
            S_AtomicStateDataBox atomicStateHostBox = getAtomicStateDataDataBox<true>();
            S_ChargeStateDataBox chargeStateHostBox = getChargeStateDataDataBox<true>();
            S_ChargeStateOrgaDataBox chargeStateOrgaHostBox = getChargeStateOrgaDataBox<true>();

            S_AtomicStateNumberOfTransitionsDataBox_UpDown<ProcClassGroup::boundFreeBased> numberTransitionsHostBox
                = getBoundFreeNumberTransitionsDataBox<true>();
            S_AtomicStateStartIndexBlockDataBox_UpDown<ProcClassGroup::boundFreeBased> startIndexBlockHostBox
                = getBoundFreeStartIndexBlockDataBox<true>();

            S_BoundFreeTransitionDataBox<s_enums::TransitionOrdering::byLowerState> transitionHostBox
                = getBoundFreeTransitionDataBox<true, s_enums::TransitionOrdering::byLowerState>();

            S_IPDIonizationStateDataBox ipdIonizationHostBox = getIPDIonizationStateDataBox<true>();

            for(CollectionIdx collectionIndexAtomicState = 0u; collectionIndexAtomicState < m_numberAtomicStates;
                ++collectionIndexAtomicState)
            {
                Idx const configNumber = atomicStateHostBox.configNumber(collectionIndexAtomicState);

                Idx const directPIStateConfigNumber = ConfigNumber::getDirectIPDIonizationState(configNumber);
                uint8_t const PIchargeState = ConfigNumber::getChargeState(directPIStateConfigNumber);

                CollectionIdx collectionIndexDirectPIState = atomicStateHostBox.findStateCollectionIndex(
                    directPIStateConfigNumber,
                    chargeStateOrgaHostBox.numberAtomicStates(PIchargeState),
                    chargeStateOrgaHostBox.startIndexBlockAtomicStates(PIchargeState));

                bool const stateExists = (collectionIndexDirectPIState < m_numberAtomicStates);

                TypeNumber const numberBoundFreeTransitions
                    = numberTransitionsHostBox.numberOfTransitionsUp(collectionIndexAtomicState);

                if(stateExists)
                    // if exists, use direct pressure ionization state
                    ipdIonizationHostBox.store(collectionIndexAtomicState, collectionIndexDirectPIState);
                else if(numberBoundFreeTransitions > 0)
                {
                    // else if at least on upward bound-free transition, use lowest delta energy transition
                    CollectionIdx const startIndexBlock
                        = startIndexBlockHostBox.startIndexBlockTransitionsUp(collectionIndexAtomicState);

                    // init with first transition
                    TypeValue lowestAbsoluteDeltaEnergy
                        = math::abs(picongpu::particles::atomicPhysics::DeltaEnergyTransition::get(
                            startIndexBlock,
                            atomicStateHostBox,
                            transitionHostBox,
                            // ionization potential depression
                            0._X,
                            chargeStateHostBox));
                    CollectionIdx index = 0u;

                    // search for upper state with energy closest to current state
                    for(CollectionIdx i = 1u; i < static_cast<CollectionIdx>(numberBoundFreeTransitions); ++i)
                    {
                        // no guarantee that delta Energy > 0, therefore we search for lowest abs(DeltaEnergy)
                        TypeValue deltaEnergy
                            = math::abs(picongpu::particles::atomicPhysics::DeltaEnergyTransition::get(
                                startIndexBlock + i,
                                atomicStateHostBox,
                                transitionHostBox,
                                // ionization potential depression
                                0._X,
                                chargeStateHostBox));

                        if(deltaEnergy < lowestAbsoluteDeltaEnergy)
                        {
                            lowestAbsoluteDeltaEnergy = deltaEnergy;
                            index = i;
                        }
                    }
                    ipdIonizationHostBox.store(collectionIndexAtomicState, startIndexBlock + index);
                }
                else
                    // disable pressure ionization
                    ipdIonizationHostBox.store(collectionIndexAtomicState, collectionIndexAtomicState);
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

            ipdIonizationStateDataBuffer.reset(new S_IPDIonizationStateDataBuffer(m_numberAtomicStates));

            // transition data, by lower state
            boundBoundTransitionDataBuffer.reset(
                new S_BoundBoundTransitionDataBuffer<s_enums::TransitionOrdering::byLowerState>(
                    m_numberBoundBoundTransitions));
            boundFreeTransitionDataBuffer.reset(
                new S_BoundFreeTransitionDataBuffer<s_enums::TransitionOrdering::byLowerState>(
                    m_numberBoundFreeTransitions));
            autonomousTransitionDataBuffer.reset(
                new S_AutonomousTransitionDataBuffer<s_enums::TransitionOrdering::byLowerState>(
                    m_numberAutonomousTransitions));

            // by upper state
            inverseBoundBoundTransitionDataBuffer.reset(
                new S_BoundBoundTransitionDataBuffer<s_enums::TransitionOrdering::byUpperState>(
                    m_numberBoundBoundTransitions));
            inverseBoundFreeTransitionDataBuffer.reset(
                new S_BoundFreeTransitionDataBuffer<s_enums::TransitionOrdering::byUpperState>(
                    m_numberBoundFreeTransitions));
            inverseAutonomousTransitionDataBuffer.reset(
                new S_AutonomousTransitionDataBuffer<s_enums::TransitionOrdering::byUpperState>(
                    m_numberAutonomousTransitions));
        }

        /** fill storage buffer directly from list
         *
         * @tparam storageDirection for which direction to store the list data, use none for non directional data
         * @tparam T_Tuple type of tuple
         * @tparam T_DataBox type of dataBox
         * @tparam T_AdditionalData tpye of additional data for use in store call to dataBox
         *
         * @param list correctly ordered list of data tuples to store
         * @param additionalData additional reference data, passed to store() call in host box
         *
         * @attention does not sync to device, must be synced externally explicitly
         */
        template<
            detail::StorageDirectionSwitch storageDirection,
            typename T_Tuple,
            typename T_DataBox,
            typename... T_AdditionalData>
        ALPAKA_FN_HOST void storeListToBuffer(
            std::list<T_Tuple>& list,
            T_DataBox hostBox,
            const T_AdditionalData... additionalData)
        {
            typename std::list<T_Tuple>::iterator iter = list.begin();

            uint32_t i = 0u;
            for(; iter != list.end(); iter++)
            {
                if constexpr(u8(storageDirection) == u8(detail::StorageDirectionSwitch::none))
                    hostBox.store(i, *iter, additionalData...);
                else if constexpr(u8(storageDirection) == u8(detail::StorageDirectionSwitch::upward))
                    hostBox.storeUp(i, *iter, additionalData...);
                else if constexpr(u8(storageDirection) == u8(detail::StorageDirectionSwitch::downward))
                    hostBox.storeDown(i, *iter, additionalData...);
                ++i;
            }
        }

        /** fill the charge state orga data buffer
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

            // read first entry to init lastChargeState
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


        /** fill the upward atomic state orga buffers for a transition group
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
         *  The transition list is assumed to consist of strict-totally ordered blocks of transitions
         *  with each block of transitions sharing the same lower state.
         *
         *  The first support points stores the current open transition blocks first transition, while the second
         *  support point advances element-wise over the transition list until it finds a transition with a lower state
         *  not equal to the first support point's lower state, i.e. the start of a new block.
         *
         *  We then note down the accumulated values for the open block, close it, and
         *  open a new block, by setting the first support point to the second support point
         *  and continue as before until we reach the end of the transition list and close the last open block.
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

    public:
        /** read input files and create/fill data boxes
         *
         * @param fileChargeData path to file containing charge state data
         * @param fileAtomicStateData path to file containing atomic state data
         * @param fileIPDIonizationData path to file containing pressure ionization data
         * @param fileBoundBoundTransitionData path to file containing bound-bound transition data
         * @param fileBoundFreeTransitionData path to file containing bound-free transition data
         * @param fileAutonomousTransitionData path to file containing autonomous transition data
         */
        AtomicData(
            std::string fileChargeStateData,
            std::string fileAtomicStateData,
            std::string fileIPDIonizationData,
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
            // boundBoundTransitions.sort(CompareTransitionTupel<TypeValue, Idx, true>());
            // boundFreeTransitions.sort(CompareTransitionTupel<TypeValue, Idx, true>());
            // autonomousTransitions.sort(CompareTransitionTupel<TypeValue, Idx, true>());

            // check user input fulfills ordering assumptions
            checkChargeStateList(chargeStates);
            checkAtomicStateList(atomicStates);
            checkTransitionList<S_BoundBoundTransitionTuple>(boundBoundTransitions);
            checkTransitionList<S_BoundFreeTransitionTuple>(boundFreeTransitions);
            checkTransitionList<S_AutonomousTransitionTuple>(autonomousTransitions);

            // initialize buffers
            initBuffers();

            // fill data buffers
            //@{
            //      charge states
            storeListToBuffer<detail::StorageDirectionSwitch::none, S_ChargeStateTuple, S_ChargeStateDataBox>(
                chargeStates,
                chargeStateDataBuffer->getHostDataBox());
            chargeStateDataBuffer->hostToDevice();

            //      atomic states
            storeListToBuffer<detail::StorageDirectionSwitch::none, S_AtomicStateTuple, S_AtomicStateDataBox>(
                atomicStates,
                atomicStateDataBuffer->getHostDataBox());
            atomicStateDataBuffer->hostToDevice();

            //      transitions
            storeListToBuffer<
                detail::StorageDirectionSwitch::none,
                S_BoundBoundTransitionTuple,
                S_BoundBoundTransitionDataBox<s_enums::TransitionOrdering::byLowerState>,
                S_AtomicStateDataBox>(
                boundBoundTransitions,
                boundBoundTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            boundBoundTransitionDataBuffer->hostToDevice();

            storeListToBuffer<
                detail::StorageDirectionSwitch::none,
                S_BoundFreeTransitionTuple,
                S_BoundFreeTransitionDataBox<s_enums::TransitionOrdering::byLowerState>,
                S_AtomicStateDataBox>(
                boundFreeTransitions,
                boundFreeTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            boundFreeTransitionDataBuffer->hostToDevice();

            storeListToBuffer<
                detail::StorageDirectionSwitch::none,
                S_AutonomousTransitionTuple,
                S_AutonomousTransitionDataBox<s_enums::TransitionOrdering::byLowerState>,
                S_AtomicStateDataBox>(
                autonomousTransitions,
                autonomousTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            autonomousTransitionDataBuffer->hostToDevice();
            //@}

            // do integrity checks on user input data
            //      check for inversions in upper lower state of transitions
            checkTransitionsForEnergyInversion(this->getBoundBoundTransitionDataBox<
                                               /*host*/ true,
                                               s_enums::TransitionOrdering::byLowerState>());
            // no check for bound-free, since bound-free transitions may reduce overall energy
            checkTransitionsForEnergyInversion(this->getAutonomousTransitionDataBox<
                                               /*host*/ true,
                                               s_enums::TransitionOrdering::byLowerState>());
            //      check all upper and lower states of transitions exist
            /// implement check, @todo Brian Marre, 2023

            if constexpr(picongpu::atomicPhysics::debug::atomicData::DEBUG_SYNC_BUFFERS_TO_HOST)
                this->hostToDevice();

            // fill orga data buffers, upward
            //@{
            //          charge state, does sync internally
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
            //@}

            // re-sort by upper state of transition
            boundBoundTransitions.sort(CompareTransitionTupel<TypeValue, T_ConfigNumber, false>());
            boundFreeTransitions.sort(CompareTransitionTupel<TypeValue, T_ConfigNumber, false>());
            autonomousTransitions.sort(CompareTransitionTupel<TypeValue, T_ConfigNumber, false>());

            // fill data data buffers, downward
            //@{
            storeListToBuffer<
                detail::StorageDirectionSwitch::none,
                S_BoundBoundTransitionTuple,
                S_BoundBoundTransitionDataBox<s_enums::TransitionOrdering::byUpperState>,
                S_AtomicStateDataBox>(
                boundBoundTransitions,
                inverseBoundBoundTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            inverseBoundBoundTransitionDataBuffer->hostToDevice();

            storeListToBuffer<
                detail::StorageDirectionSwitch::none,
                S_BoundFreeTransitionTuple,
                S_BoundFreeTransitionDataBox<s_enums::TransitionOrdering::byUpperState>,
                S_AtomicStateDataBox>(
                boundFreeTransitions,
                inverseBoundFreeTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            inverseBoundFreeTransitionDataBuffer->hostToDevice();

            storeListToBuffer<
                detail::StorageDirectionSwitch::none,
                S_AutonomousTransitionTuple,
                S_AutonomousTransitionDataBox<s_enums::TransitionOrdering::byUpperState>,
                S_AtomicStateDataBox>(
                autonomousTransitions,
                inverseAutonomousTransitionDataBuffer->getHostDataBox(),
                atomicStateDataBuffer->getHostDataBox());
            inverseAutonomousTransitionDataBuffer->hostToDevice();

            // fill orga buffers, downward
            //@{
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
            //@}

            if constexpr(picongpu::atomicPhysics::debug::atomicData::DEBUG_SYNC_BUFFERS_TO_HOST)
                this->hostToDevice();

            // fill ipdIonizationStateDataBuffer
            if(fileIPDIonizationData.empty())
            {
                std::cout
                    << "no user provided presssure ioniation state file, using automatic pressure ionization states"
                    << std::endl;
                fillIPDIonizationDataAuto();
            }
            else
            {
                // read in user provided file
                std::list<S_IPDIonizationStateTuple> ipdIonizationStateList
                    = readIPDIonizationStates(fileIPDIonizationData);

                // check ordering assumptions and run integrity checks
                checkIPDIonizationList(ipdIonizationStateList);
                fillIPDIonizationDataFromList(ipdIonizationStateList);
            }
            //      sync to device
            ipdIonizationStateDataBuffer->hostToDevice();

            if constexpr(picongpu::atomicPhysics::debug::atomicData::DEBUG_SYNC_BUFFERS_TO_HOST)
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

            // ipdIonizationState data
            ipdIonizationStateDataBuffer->hostToDevice();

            // transition data
            boundBoundTransitionDataBuffer->hostToDevice();
            boundFreeTransitionDataBuffer->hostToDevice();
            autonomousTransitionDataBuffer->hostToDevice();

            // inverse transition data
            inverseBoundBoundTransitionDataBuffer->hostToDevice();
            inverseBoundFreeTransitionDataBuffer->hostToDevice();
            inverseAutonomousTransitionDataBuffer->hostToDevice();
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

            // ipdIonizationState data
            ipdIonizationStateDataBuffer->deviceToHost();

            // transition data
            boundBoundTransitionDataBuffer->deviceToHost();
            boundFreeTransitionDataBuffer->deviceToHost();
            autonomousTransitionDataBuffer->deviceToHost();

            // inverse transition data
            inverseBoundBoundTransitionDataBuffer->deviceToHost();
            inverseBoundFreeTransitionDataBuffer->deviceToHost();
            inverseAutonomousTransitionDataBuffer->deviceToHost();
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

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_IPDIonizationStateDataBox getIPDIonizationStateDataBox()
        {
            if constexpr(hostData)
                return ipdIonizationStateDataBuffer->getHostDataBox();
            else
                return ipdIonizationStateDataBuffer->getDeviceDataBox();

            ALPAKA_UNREACHABLE(ipdIonizationStateDataBuffer->getHostDataBox());
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
        template<bool hostData, s_enums::TransitionOrdering T_TransitionOrdering>
        S_BoundBoundTransitionDataBox<T_TransitionOrdering> getBoundBoundTransitionDataBox()
        {
            constexpr bool byLowerState = (u8(T_TransitionOrdering) == u8(s_enums::TransitionOrdering::byLowerState));

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
        template<bool hostData, s_enums::TransitionOrdering T_TransitionOrdering>
        S_BoundFreeTransitionDataBox<T_TransitionOrdering> getBoundFreeTransitionDataBox()
        {
            constexpr bool byLowerState = (u8(T_TransitionOrdering) == u8(s_enums::TransitionOrdering::byLowerState));

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
        template<bool hostData, s_enums::TransitionOrdering T_TransitionOrdering>
        S_AutonomousTransitionDataBox<T_TransitionOrdering> getAutonomousTransitionDataBox()
        {
            constexpr bool byLowerState = (u8(T_TransitionOrdering) == u8(s_enums::TransitionOrdering::byLowerState));

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

} // namespace picongpu::particles::atomicPhysics::atomicData
