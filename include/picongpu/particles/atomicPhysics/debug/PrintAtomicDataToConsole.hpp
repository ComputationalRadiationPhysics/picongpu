/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2024-2025 Brian Marre
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
#include "picongpu/particles/atomicPhysics/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClassGroup.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace picongpu::particles::atomicPhysics::debug
{
    namespace s_enums = picongpu::particles::atomicPhysics::enums;

    //! print active process classes
    template<typename T_AtomicData>
    ALPAKA_FN_HOST void printProcessConfiguration()
    {
        std::cout << "process configuration:" << std::endl;
        std::cout << "\t Electronic Excitation:    " << ((T_AtomicData::switchElectronicExcitation) ? "true" : "false")
                  << std::endl;
        std::cout << "\t Electronic DeExcitation:  "
                  << ((T_AtomicData::switchElectronicDeexcitation) ? "true" : "false") << std::endl;
        std::cout << "\t Spontaneous DeExcitation: "
                  << ((T_AtomicData::switchSpontaneousDeexcitation) ? "true" : "false") << std::endl;
        std::cout << "\t Electronic Ionization:    " << ((T_AtomicData::switchElectronicIonization) ? "true" : "false")
                  << std::endl;
        std::cout << "\t Autonomous ionization:    " << ((T_AtomicData::switchAutonomousIonization) ? "true" : "false")
                  << std::endl;
        std::cout << "\t Field Ionization:         " << ((T_AtomicData::switchFieldIonization) ? "true" : "false")
                  << std::endl;
    }

    template<typename T_AtomicData>
    ALPAKA_FN_HOST std::unique_ptr<T_AtomicData> printBasicStatistics(std::unique_ptr<T_AtomicData> atomicData)
    {
        uint32_t const numberAtomicStates = atomicData->getNumberAtomicStates();
        uint32_t const numberBoundBoundTransitions = atomicData->getNumberBoundBoundTransitions();
        uint32_t const numberBoundFreeTransitions = atomicData->getNumberBoundFreeTransitions();
        uint32_t const numberAutonomousTransitions = atomicData->getNumberAutonomousTransitions();
        constexpr auto atomicNumber = static_cast<uint16_t>(T_AtomicData::ConfigNumber::atomicNumber);

        std::cout << "Basic Statistics:" << std::endl;
        std::cout << "AtomicNumber: " << atomicNumber << "(#s " << numberAtomicStates << ", #b "
                  << numberBoundBoundTransitions << ", #f " << numberBoundFreeTransitions << ", #a "
                  << numberAutonomousTransitions << ")" << std::endl;

        return atomicData;
    }

    //! print header of chargeState Data
    ALPAKA_FN_HOST inline void printChargeStateDataHeader()
    {
        std::cout << "ChargeState Data" << std::endl;
        std::cout << "index : E_ionization[eV] [#AtomicStates, startIndexBlock], "
                  << "b:[#TransitionsUp / #TransitionsDown], "
                  << "f:[#TransitionsUp / #TransitionsDown], "
                  << "a:[#TransitionsDown]" << std::endl;
    }

    //! print all data stored by charge state
    template<typename T_AtomicData>
    ALPAKA_FN_HOST void printByChargeStateStoredData(
        uint8_t const chargeState,
        auto chargeStateDataBox,
        auto chargeStateOrgaBox)
    {
        if(chargeState == T_AtomicData::ConfigNumber::atomicNumber)
        {
            std::cout << std::setw(3) << static_cast<uint16_t>(T_AtomicData::ConfigNumber::atomicNumber) << ":"
                      << "      na, [" << std::setw(4)
                      << chargeStateOrgaBox.numberAtomicStates(T_AtomicData::ConfigNumber::atomicNumber) << ", "
                      << std::setw(4)
                      << chargeStateOrgaBox.startIndexBlockAtomicStates(T_AtomicData::ConfigNumber::atomicNumber)
                      << "],";
        }
        else
        {
            std::cout << std::setw(3) << static_cast<uint16_t>(chargeState) << ":" << std::setw(8)
                      << chargeStateDataBox.ionizationEnergy(chargeState) << ", [" << std::setw(4)
                      << chargeStateOrgaBox.numberAtomicStates(chargeState) << ", " << std::setw(4)
                      << chargeStateOrgaBox.startIndexBlockAtomicStates(chargeState) << "],";
        }
    }

    /** debug only, get number of Transitions of given charge state
     *
     * @tparam T_up true =^= sum over upward transitions, false =^= sum over downward transitions
     *
     * @param numberAtomicStatesOfChargeState number of entries of the charge state's block of atomic states in the
     * list of atomic states
     * @param startIndexBlockAtomicStates first index of the charge state's block of atomic states in the list of
     * atomic states
     */
    template<bool T_up>
    ALPAKA_FN_HOST uint32_t getNumberTransitionsOfChargeState(
        uint32_t const numberAtomicStatesOfChargeState,
        uint32_t const startIndexBlockAtomicStatesOfChargeState,
        auto numberTransitionsBox)
    {
        uint32_t numberTransitions = 0u;
        for(uint32_t state = 0u; state < numberAtomicStatesOfChargeState; state++)
        {
            uint32_t stateCollectionIndex = state + startIndexBlockAtomicStatesOfChargeState;
            if constexpr(T_up)
            {
                numberTransitions += numberTransitionsBox.numberOfTransitionsUp(stateCollectionIndex);
            }
            else
            {
                numberTransitions += numberTransitionsBox.numberOfTransitionsDown(stateCollectionIndex);
            }
        }
        return numberTransitions;
    }

    /** print the number of transitions of the charge state
     *
     * @param chargeState
     * @param chargeStateOrgaBox host data box giving access to the organizational data of each charge state
     * @param boundBoundNumberTransitionsBox host data box giving access to the number of up- and downward bound-bound
     *  transitions of each atomic state
     * @param boundFreeNumberTransitionsBox host data box giving access to the number of up- and downward bound-free
     *  transitions of each atomic state
     * @param autonomousNumberTransitionsBox host data box giving access to the number of downward autonomous
     * transitions of each atomic state
     */
    ALPAKA_FN_HOST inline void printNumberOfTransitionsOfChargeState(
        uint8_t const chargeState,
        auto chargeStateOrgaBox,
        auto boundBoundNumberTransitionsBox,
        auto boundFreeNumberTransitionsBox,
        auto autonomousNumberTransitionsBox)
    {
        uint32_t numberAtomicStatesOfChargeState = chargeStateOrgaBox.numberAtomicStates(chargeState);
        uint32_t startIndexBlockOfChargeState = chargeStateOrgaBox.startIndexBlockAtomicStates(chargeState);

        std::cout << " b:[" << std::setw(5)
                  << getNumberTransitionsOfChargeState<true>(
                         numberAtomicStatesOfChargeState,
                         startIndexBlockOfChargeState,
                         boundBoundNumberTransitionsBox)
                  << " / " << std::setw(5)
                  << getNumberTransitionsOfChargeState<false>(
                         numberAtomicStatesOfChargeState,
                         startIndexBlockOfChargeState,
                         boundBoundNumberTransitionsBox)
                  << "], ";
        std::cout << "f:[" << std::setw(5)
                  << getNumberTransitionsOfChargeState<true>(
                         numberAtomicStatesOfChargeState,
                         startIndexBlockOfChargeState,
                         boundFreeNumberTransitionsBox)
                  << " / " << std::setw(5)
                  << getNumberTransitionsOfChargeState<false>(
                         numberAtomicStatesOfChargeState,
                         startIndexBlockOfChargeState,
                         boundFreeNumberTransitionsBox)
                  << "], ";
        std::cout << "a:[" << std::setw(5)
                  << getNumberTransitionsOfChargeState<false>(
                         numberAtomicStatesOfChargeState,
                         startIndexBlockOfChargeState,
                         autonomousNumberTransitionsBox)
                  << "]";
    }

    //! print header of atomic state data
    ALPAKA_FN_HOST inline void printAtomicStateDataHeader()
    {
        std::cout << "AtomicState Data" << std::endl;
        std::cout << "index : [ConfigNumber, chargeState, levelVector]: E_overGround, screenedCharge, multiplicity, "
                     "IPDIonizationState[index, chargeState, (dE to state [eV]), levelVector]"
                  << std::endl;
        std::cout << "\t b/f/a: [#TransitionsUp/]#TransitionsDown, [startIndexUp/]startIndexDown" << std::endl;
    }

    //! print all by atomic states stored data
    template<typename T_AtomicData>
    ALPAKA_FN_HOST std::unique_ptr<T_AtomicData> printAtomicStateStoredData(
        uint32_t const stateCollectionIndex,
        std::unique_ptr<T_AtomicData> atomicData)
    {
        auto chargeStateDataBox = atomicData->template getChargeStateDataDataBox<true>(); // true: get hostDataBox
        auto atomicStateDataBox = atomicData->template getAtomicStateDataDataBox<true>();
        auto ipdIonizationStateDataBox = atomicData->template getIPDIonizationStateDataBox<true>();

        auto boundBoundNumberTransitionsBox = atomicData->template getBoundBoundNumberTransitionsDataBox<true>();
        auto boundFreeNumberTransitionsBox = atomicData->template getBoundFreeNumberTransitionsDataBox<true>();
        auto autonomousNumberTransitionsBox = atomicData->template getAutonomousNumberTransitionsDataBox<true>();

        auto boundBoundStartIndexBox = atomicData->template getBoundBoundStartIndexBlockDataBox<true>();
        auto boundFreeStartIndexBox = atomicData->template getBoundFreeStartIndexBlockDataBox<true>();
        auto autonomousStartIndexBox = atomicData->template getAutonomousStartIndexBlockDataBox<true>();

        using S_ConfigNumber = stateRepresentation::
            ConfigNumber<uint64_t, T_AtomicData::ConfigNumber::numberLevels, T_AtomicData::ConfigNumber::atomicNumber>;

        auto const stateConfigNumber = atomicStateDataBox.configNumber(stateCollectionIndex);
        auto const chargeState = static_cast<uint16_t>(S_ConfigNumber::getChargeState(stateConfigNumber));
        auto const stateLevelVector = precisionCast<uint16_t>(S_ConfigNumber::getLevelVector(stateConfigNumber));

        auto const ipdIonizationStateCollectionIndex
            = ipdIonizationStateDataBox.ipdIonizationState(stateCollectionIndex);
        auto const chargeStateIPDIonizationVector = static_cast<uint16_t>(
            S_ConfigNumber::getChargeState(atomicStateDataBox.configNumber(ipdIonizationStateCollectionIndex)));
        auto const levelVectorIPDIonizationState = precisionCast<uint16_t>(
            S_ConfigNumber::getLevelVector(atomicStateDataBox.configNumber(ipdIonizationStateCollectionIndex)));
        auto const energyToIPDIonizationState
            = atomicStateDataBox.energy(ipdIonizationStateCollectionIndex)
              + DeltaEnergyTransition::template ionizationEnergy<s_enums::ProcessClassGroup::boundFreeBased, float_X>(
                  chargeState,
                  chargeStateIPDIonizationVector,
                  0._X,
                  chargeStateDataBox)
              - atomicStateDataBox.energy(stateCollectionIndex);

        auto const multiplicity = static_cast<uint64_t>(atomicStateDataBox.multiplicity(stateCollectionIndex));

        std::cout << "\t" << stateCollectionIndex << " : [" << stateConfigNumber << ", " << chargeState << ", "
                  << stateLevelVector.toString(",", "()") << "]: " << atomicStateDataBox.energy(stateCollectionIndex)
                  << ", " << atomicStateDataBox.screenedCharge(stateCollectionIndex) << ", " << multiplicity << ",\t["
                  << ipdIonizationStateCollectionIndex << ", " << static_cast<uint16_t>(chargeStateIPDIonizationVector)
                  << ", " << energyToIPDIonizationState << ", " << levelVectorIPDIonizationState.toString(",", "()")
                  << "]" << std::endl;
        std::cout << "\t\t b: " << boundBoundNumberTransitionsBox.numberOfTransitionsUp(stateCollectionIndex) << "/"
                  << boundBoundNumberTransitionsBox.numberOfTransitionsDown(stateCollectionIndex) << ", "
                  << boundBoundStartIndexBox.startIndexBlockTransitionsUp(stateCollectionIndex) << "/"
                  << boundBoundStartIndexBox.startIndexBlockTransitionsDown(stateCollectionIndex) << std::endl;
        std::cout << "\t\t f: " << boundFreeNumberTransitionsBox.numberOfTransitionsUp(stateCollectionIndex) << "/"
                  << boundFreeNumberTransitionsBox.numberOfTransitionsDown(stateCollectionIndex) << ", "
                  << boundFreeStartIndexBox.startIndexBlockTransitionsUp(stateCollectionIndex) << "/"
                  << boundFreeStartIndexBox.startIndexBlockTransitionsDown(stateCollectionIndex) << std::endl;
        std::cout << "\t\t a: " << autonomousNumberTransitionsBox.numberOfTransitionsDown(stateCollectionIndex) << ", "
                  << autonomousStartIndexBox.startIndexBlockTransitionsDown(stateCollectionIndex) << std::endl;
        return atomicData;
    }

    /** debug only, write atomic data to console
     *
     * @attention must be called serially!
     */
    template<typename T_AtomicData, bool T_printTransitionData, bool T_printInverseTransitions>
    ALPAKA_FN_HOST std::unique_ptr<T_AtomicData> printAtomicDataToConsole(std::unique_ptr<T_AtomicData> atomicData)
    {
        std::cout << std::endl << "**AtomicData DEBUG Output**" << std::endl;

        printProcessConfiguration<T_AtomicData>();
        atomicData = printBasicStatistics(std::move(atomicData));

        // chargeStates
        auto chargeStateDataBox = atomicData->template getChargeStateDataDataBox</* get hostDataBox */ true>();
        auto chargeStateOrgaBox = atomicData->template getChargeStateOrgaDataBox<true>();

        auto boundBoundNumberTransitionsBox = atomicData->template getBoundBoundNumberTransitionsDataBox<true>();
        auto boundFreeNumberTransitionsBox = atomicData->template getBoundFreeNumberTransitionsDataBox<true>();
        auto autonomousNumberTransitionsBox = atomicData->template getAutonomousNumberTransitionsDataBox<true>();

        printChargeStateDataHeader();
        for(uint8_t chargeState = 0u; chargeState < (T_AtomicData::ConfigNumber::atomicNumber + 1u); chargeState++)
        {
            printByChargeStateStoredData<T_AtomicData>(chargeState, chargeStateDataBox, chargeStateOrgaBox);

            printNumberOfTransitionsOfChargeState(
                chargeState,
                chargeStateOrgaBox,
                boundBoundNumberTransitionsBox,
                boundFreeNumberTransitionsBox,
                autonomousNumberTransitionsBox);
            std::cout << std::endl;
        }

        printAtomicStateDataHeader();
        auto const numberAtomicStates = atomicData->getNumberAtomicStates();
        for(uint32_t stateCollectionIndex = 0u; stateCollectionIndex < numberAtomicStates; stateCollectionIndex++)
        {
            atomicData = printAtomicStateStoredData(stateCollectionIndex, std::move(atomicData));
        }

        // AtomicState data
        auto atomicStateDataBox = atomicData->template getAtomicStateDataDataBox</* get hostDataBox */ true>();

        // transitionData
        if constexpr(T_printTransitionData)
        {
            // bound-bound transitions
            auto boundBoundTransitionDataBox
                = atomicData
                      ->template getBoundBoundTransitionDataBox<true, s_enums::TransitionOrdering::byLowerState>();
            auto const numberBoundBoundTransitions = atomicData->getNumberBoundBoundTransitions();

            std::cout << "bound-bound transition" << std::endl;
            std::cout << "index (low, up), dE, C, A, \"Gaunt\"( <1>, <2>, ...)" << std::endl;

            for(uint32_t i = 0u; i < numberBoundBoundTransitions; i++)
            {
                std::cout << i << "\t(" << boundBoundTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << boundBoundTransitionDataBox.upperStateCollectionIndex(i) << ")"
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(i, atomicStateDataBox, boundBoundTransitionDataBox)
                          << ",\tC: " << boundBoundTransitionDataBox.collisionalOscillatorStrength(i)
                          << ",\tA: " << boundBoundTransitionDataBox.absorptionOscillatorStrength(i) << "\t\"Gaunt\"( "
                          << boundBoundTransitionDataBox.cxin1(i) << ", " << boundBoundTransitionDataBox.cxin2(i)
                          << ", " << boundBoundTransitionDataBox.cxin3(i) << ", "
                          << boundBoundTransitionDataBox.cxin4(i) << ", " << boundBoundTransitionDataBox.cxin5(i)
                          << " )" << std::endl;
            }

            // bound-free transitions
            auto boundFreeTransitionDataBox
                = atomicData
                      ->template getBoundFreeTransitionDataBox<true, s_enums::TransitionOrdering::byLowerState>();
            auto const numberBoundFreeTransitions = atomicData->getNumberBoundFreeTransitions();

            std::cout << "bound-free transition" << std::endl;
            std::cout << "index (low, up), dE, Coeff( <1>, <2>, ...)" << std::endl;

            for(uint32_t i = 0u; i < numberBoundFreeTransitions; i++)
            {
                std::cout << i << "\t(" << boundFreeTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << boundFreeTransitionDataBox.upperStateCollectionIndex(i) << ")"
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(
                                 i,
                                 atomicStateDataBox,
                                 boundFreeTransitionDataBox,
                                 // eV, ionization potential depression is dynamics dependent, therefore set to zero
                                 // for debug
                                 0._X,
                                 chargeStateDataBox)
                          << "\tCoeff(" << boundFreeTransitionDataBox.cxin1(i) << ", "
                          << boundFreeTransitionDataBox.cxin2(i) << ", " << boundFreeTransitionDataBox.cxin3(i) << ", "
                          << boundFreeTransitionDataBox.cxin4(i) << ", " << boundFreeTransitionDataBox.cxin5(i) << ", "
                          << boundFreeTransitionDataBox.cxin6(i) << ", " << boundFreeTransitionDataBox.cxin7(i) << ", "
                          << boundFreeTransitionDataBox.cxin8(i) << ")" << std::endl;
            }

            // autonomous transitions
            auto autonomousTransitionDataBox
                = atomicData
                      ->template getAutonomousTransitionDataBox<true, s_enums::TransitionOrdering::byLowerState>();
            auto const numberAutonomousTransitions = atomicData->getNumberAutonomousTransitions();

            std::cout << "autonomous transitions" << std::endl;
            std::cout << "index (low, up), dE, rate [1/Dt_PIC]" << std::endl;

            for(uint32_t i = 0u; i < numberAutonomousTransitions; i++)
            {
                std::cout << i << "\t(" << autonomousTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << autonomousTransitionDataBox.upperStateCollectionIndex(i) << ") "
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(
                                 i,
                                 atomicStateDataBox,
                                 autonomousTransitionDataBox,
                                 // eV, ionization potential depression is dynamics dependent, therefore set to zero
                                 // for debug
                                 0._X,
                                 chargeStateDataBox)
                          << ",\trate: " << autonomousTransitionDataBox.rate(i) << std::endl;
            }
            std::cout << std::endl;
        }

        // inverse transitionData
        if constexpr(T_printInverseTransitions)
        {
            // bound-bound transitions
            auto boundBoundTransitionDataBox
                = atomicData
                      ->template getBoundBoundTransitionDataBox<true, s_enums::TransitionOrdering::byUpperState>();
            auto const numberBoundBoundTransitions = atomicData->getNumberBoundBoundTransitions();

            std::cout << "inverse bound-bound transition" << std::endl;
            std::cout << "index (low, up), dE, C, A, \"Gaunt\"( <1>, <2>, ...)" << std::endl;

            for(uint32_t i = 0u; i < numberBoundBoundTransitions; i++)
            {
                std::cout << i << "\t(" << boundBoundTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << boundBoundTransitionDataBox.upperStateCollectionIndex(i) << ")"
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(i, atomicStateDataBox, boundBoundTransitionDataBox)
                          << ",\tC: " << boundBoundTransitionDataBox.collisionalOscillatorStrength(i)
                          << ",\tA: " << boundBoundTransitionDataBox.absorptionOscillatorStrength(i) << "\t\"Gaunt\"("
                          << boundBoundTransitionDataBox.cxin1(i) << ", " << boundBoundTransitionDataBox.cxin2(i)
                          << ", " << boundBoundTransitionDataBox.cxin3(i) << ", "
                          << boundBoundTransitionDataBox.cxin4(i) << ", " << boundBoundTransitionDataBox.cxin5(i)
                          << ")" << std::endl;
            }

            // bound-free transitions
            auto boundFreeTransitionDataBox
                = atomicData
                      ->template getBoundFreeTransitionDataBox<true, s_enums::TransitionOrdering::byUpperState>();
            auto const numberBoundFreeTransitions = atomicData->getNumberBoundFreeTransitions();

            std::cout << "inverse bound-free transition" << std::endl;
            std::cout << "index (low, up), dE, Coeff( <1>, <2>, ...)" << std::endl;

            for(uint32_t i = 0u; i < numberBoundFreeTransitions; i++)
            {
                std::cout << i << "\t(" << boundFreeTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << boundFreeTransitionDataBox.upperStateCollectionIndex(i) << ")"
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(
                                 i,
                                 atomicStateDataBox,
                                 boundFreeTransitionDataBox,
                                 // eV, ionization potential depression is dynamics dependent, therefore set to zero
                                 // for debug
                                 0._X,
                                 chargeStateDataBox)
                          << "\tCoeff(" << boundFreeTransitionDataBox.cxin1(i) << ", "
                          << boundFreeTransitionDataBox.cxin2(i) << ", " << boundFreeTransitionDataBox.cxin3(i) << ", "
                          << boundFreeTransitionDataBox.cxin4(i) << ", " << boundFreeTransitionDataBox.cxin5(i) << ", "
                          << boundFreeTransitionDataBox.cxin6(i) << ", " << boundFreeTransitionDataBox.cxin7(i) << ", "
                          << boundFreeTransitionDataBox.cxin8(i) << ")" << std::endl;
            }

            // autonomous transitions
            auto autonomousTransitionDataBox
                = atomicData
                      ->template getAutonomousTransitionDataBox<true, s_enums::TransitionOrdering::byUpperState>();
            auto const numberAutonomousTransitions = atomicData->getNumberAutonomousTransitions();

            std::cout << "inverse autonomous transitions" << std::endl;
            std::cout << "index (low, up), dE, rate" << std::endl;

            for(uint32_t i = 0u; i < numberAutonomousTransitions; i++)
            {
                std::cout << i << "\t(" << autonomousTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << autonomousTransitionDataBox.upperStateCollectionIndex(i) << ") "
                          << ", dE: "
                          << DeltaEnergyTransition::get(
                                 i,
                                 atomicStateDataBox,
                                 autonomousTransitionDataBox,
                                 // eV, ionization potential depression is dynamics dependent, therefore set to zero
                                 // for debug
                                 0._X,
                                 chargeStateDataBox)
                          << ", rate: " << autonomousTransitionDataBox.rate(i) << std::endl;
            }
            std::cout << std::endl;
        }
        return atomicData;
    }
} // namespace picongpu::particles::atomicPhysics::debug
