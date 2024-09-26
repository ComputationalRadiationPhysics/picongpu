/* Copyright 2023-2024 Brian Marre, Axel Huebl
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

#include "picongpu/defines.hpp" // need atomicPhysics_Debug.param
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/CollisionalRate.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/Multiplicities.hpp"
#include "picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cstdint>

/** @file implements calculation of rates for bound-free collisional atomic physics transitions
 *
 * this includes ionization due to:
 *  - free electron interaction
 *  @todo photon ionization based processes, Brian Marre, 2023
 *  @todo recombination processes Brian Marre, 2023
 *
 * based on the rate calculation of FLYCHK, as extracted by Axel Huebl in the original
 *  flylite prototype.
 *
 * References:
 * - Axel Huebl
 *  first flylite prototype, not published
 *
 * - A. Burgess, M.C. Chidichimo.
 * "Electron impact ionization of complex ions."
 * Mon. Not. R. astr. Soc. 203, 1269-1280 (1983)
 *   based on the works of
 *     W. Lotz
 *     Zeitschrift fuer Physik 216, 241-247 (1968)
 */

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    /** compilation of static rate- and crossSection-calc. methods for all processes
     * working on bound-free transition data
     *
     * @tparam T_numberLevels maximum principal quantum number of atomic states of species
     * @tparam T_debug activate debug print to console output
     *
     * @attention atomic data box input data is assumed to be in eV
     */
    template<uint8_t T_numberLevels, bool T_debug = false>
    struct BoundFreeCollisionalTransitionRates
    {
    private:
        /** beta factor from Burgess and Chidichimo(1983)
         *
         * @param Z screened charge of the ion, [e]
         *
         * @return unitless
         */
        HDINLINE static float_X betaFactor(float_X const Z)
        {
            float_X const x = (100._X * Z + 91._X) / (4._X * Z + 3._X);
            return 0.25_X * (math::sqrt(x) - 5._X);
        }

        /** w factor from Burgess and Chidichimo(1983)
         *
         * @param U (kinetic energy of interacting electron)/(ionization potential of initial level), unitless
         * @param beta beta factor from
         *
         * @return unitless
         */
        HDINLINE static float_64 wFactor(float_X const U, float_X const beta)
        {
            return math::pow(static_cast<float_64>(math::log(U)), static_cast<float_64>(beta / U));
        }

    public:
        /** ionization cross section for a given bound-free transition and electron energy
         *
         * @tparam T_ChargeStateDataBox instantiated type of dataBox
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundFreeTransitionDataBox instantiated type of dataBox
         *
         * @param energyElectron kinetic energy of interacting free electron(/electron bin), eV
         * @param ionizationPotentialDepression in eV
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 10^6*b = 10^(-22)m^2; (b == barn = 10^(-28) m^2)
         *
         * @attention assumes that chargeStateDataBox, atomicStateDataBox and boundFreeTransitionDataBox
         *      belong to the same AtomicData instance
         */
        template<typename T_ChargeStateDataBox, typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X collisionalIonizationCrossSection(
            // eV
            float_X const energyElectron,
            // eV
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            /// @todo provide as constexpr with one of the dataBoxes?, Brian Marre, 2023
            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;
            using LevelVector = pmacc::math::Vector<uint8_t, T_numberLevels>;

            uint32_t const upperStateClctIdx
                = boundFreeTransitionDataBox.upperStateCollectionIndex(transitionCollectionIndex);
            uint32_t const lowerStateClctIdx
                = boundFreeTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);

            float_64 const combinatorialFactor
                = static_cast<float_64>(boundFreeTransitionDataBox.multiplicity(transitionCollectionIndex));

            auto const upperStateConfigNumber = atomicStateDataBox.configNumber(upperStateClctIdx);
            auto const lowerStateConfigNumber = atomicStateDataBox.configNumber(lowerStateClctIdx);

            // eV
            float_X const energyDifference = picongpu::particles::atomicPhysics::DeltaEnergyTransition::get(
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox,
                ionizationPotentialDepression,
                chargeStateDataBox);

            if constexpr(picongpu::atomicPhysics::debug::rateCalculation::DEBUG_CHECKS)
                if(S_ConfigNumber::getChargeState(upperStateConfigNumber)
                   < S_ConfigNumber::getChargeState(lowerStateConfigNumber))
                {
                    printf("atomicPhysics ERROR: upper and lower state inverted in "
                           "collisionalIonizationCrossSection() call\n");
                    return 0._X;
                }

            float_X result = 0._X;
            if(energyDifference <= energyElectron)
            {
                constexpr float_64 C = 2.3; // a fitting factor well suited for Z >= 2, unitless
                constexpr float_64 a0 = sim.si.getBohrRadius(); // m
                constexpr float_64 E_R = sim.si.conv().joule2eV(sim.si.getRydbergEnergy()); // eV
                constexpr float_64 scalingConstant
                    = C * picongpu::PI * pmacc::math::cPow(a0, 2u) / 1e-22 * pmacc::math::cPow(E_R, 2u);
                // 10^6*b * eV^2
                // m^2 / (m^2/10^6*b) * (eV)^2= m^2/m^2 * 10^6*b * eV^2

                uint8_t const lowerStateChargeState = S_ConfigNumber::getChargeState(lowerStateConfigNumber);

                /// @todo replace with screenedCharge(upperStateChargeState)?, Brian Marre, 2023
                float_X const screenedCharge = chargeStateDataBox.screenedCharge(lowerStateChargeState) - 1._X; // [e]

                float_X const U = energyElectron / energyDifference; // unitless
                float_X const beta = betaFactor(screenedCharge); // unitless
                float_64 const w = wFactor(U, beta); // unitless
                float_X const crossSection = static_cast<float_X>(
                    scalingConstant * combinatorialFactor
                    / static_cast<float_64>(pmacc::math::cPow(energyDifference, 2u)) / static_cast<float_64>(U)
                    * math::log(static_cast<float_64>(U)) * w); // [1e6*b]
                // 1e6*b * (eV)^2 * unitless / (eV)^2 / unitless * log(unitless) * unitless
                if(crossSection > 0._X)
                    result = crossSection;
            }

            return result;
        }

        /** rate for collisional bound-free (ionization) transition of ion with free electron bin
         *
         * uses second order integration(bin middle)
         *
         * @todo implement higher order integrations, Brian Marre, 2023
         *
         * @tparam T_ChargeStateDataBox instantiated type of dataBox
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundFreeTransitionDataBox instantiated type of dataBox
         *
         * @param energyElectron kinetic energy of interacting electron(/electron bin), [eV]
         * @param energyElectronBinWidth energy width of electron bin, [eV]
         * @param densityElectrons [1/(m^3 * eV)], local superCell number density of electrons in this bin
         * @param ionizationPotentialDepression eV
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 1/sim.unit.time()
         */
        template<typename T_ChargeStateDataBox, typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X rateCollisionalIonizationTransition(
            // eV
            float_X const energyElectron,
            // eV
            float_X const energyElectronBinWidth,
            // 1/(sim.unit.length()^3*eV)
            float_X const densityElectrons,
            // eV
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            float_X sigma = collisionalIonizationCrossSection(
                // eV
                energyElectron,
                ionizationPotentialDepression,
                transitionCollectionIndex,
                chargeStateDataBox,
                atomicStateDataBox,
                boundFreeTransitionDataBox); // [1e6*b]

            return picongpu::particles2::atomicPhysics::rateCalculation::collisionalRate(
                energyElectron,
                energyElectronBinWidth,
                densityElectrons,
                sigma);
        }

        /// @todo radiativeIonizationCrossSection, Scofield+Kramer
        /// @todo rateRadiativeIonization

        /// @todo spontaneousRadiativeRecombinationCrossSection
        /// @todo rateSpontaneousRadiaitveRecombination

        /// @todo threeBodyRecombinationCrossSection
        /// @todo rateCollisionalThreeBodyRecombination
        /// @todo stimulatedRecombinationCrossSection
        /// @todo rateStimulatedRecombination
    };
} // namespace picongpu::particles::atomicPhysics::rateCalculation
