/* Copyright 2022-2023 Brian Marre, Axel Huebl
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
/* need the following .param files
 *  - atomicPhysics_Debug.param      debug check switches
 *  - physicalConstants.param         physical constants, namespace picongpu::SI
 *  - unit.param                      unit of time for normalization
 */

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicData.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/CollisionalRate.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/Multiplicities.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cmath>
#include <cstdint>


/** @file implements calculation of rates for bound-bound atomic physics transitions
 *
 * this includes:
 *  - electron-interaction base de-/excitation
 *  - spontaneous photon emission
 *  @todo photon interaction based processes, Brian Marre, 2022
 *
 * based on the rate calculation of FLYCHK, as extracted by Axel Huebl in the original
 *  flylite prototype.
 *
 * References:
 * - Axel Huebl
 *  first flylite prototype, not published
 *
 * - R. Mewe.
 *  "Interpolation formulae for the electron impact excitation of ions in
 *  the H-, He-, Li-, and Ne-sequences."
 *  Astronomy and Astrophysics 20, 215 (1972)
 *
 * - H.-K. Chung, R.W. Lee, M.H. Chen.
 *  "A fast method to generate collisional excitation cross-sections of
 *  highly charged ions in a hot dense matter"
 *  High Energy Density Physics 3, 342-352 (2007)
 *
 * - and https://en.wikipedia.org/wiki/Einstein_coefficients for the SI version
 */

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    /** compilation of static rate- and crossSection-calc. methods for all processes
     * working on bound-bound transition data
     *
     * @tparam T_numberLevels maximum principal quantum number of atomic states of ion species
     *
     * @attention atomic data box input data is assumed to be in eV
     */
    template<uint8_t T_numberLevels>
    struct BoundBoundTransitionRates
    {
    private:
        template<typename T_Type>
        HDINLINE static bool relativeTest(T_Type trueValue, T_Type testValue, T_Type errorLimit)
        {
            return math::abs((testValue - trueValue) / trueValue) > errorLimit;
        }

        /** gaunt factor suppression of cross sections
         *
         * @param U = energyElectron / energyDifference, unitless
         * @param collectionIndexTransition internal index of transition in transitionDataBox
         *      use findIndexTransition method of atomicDataBox and screen for not found value
         * @param boundBoundTransitionDataBox data box giving access to bound-bound transition data
         *
         * @attention no range check for indexTransition outside debug compile, invalid memory access otherwise
         *
         * @return unit: unitless
         */
        template<typename T_BoundBoundTransitionDataBox>
        HDINLINE static float_X gauntFactor(
            float_X const U, // unitless
            uint32_t const collectionIndexTransition,
            T_BoundBoundTransitionDataBox const boundBoundTransitionDataBox)
        {
            // get gaunt coefficients, unitless
            float_X const A = boundBoundTransitionDataBox.cxin1(collectionIndexTransition);
            float_X const B = boundBoundTransitionDataBox.cxin2(collectionIndexTransition);
            float_X const C = boundBoundTransitionDataBox.cxin3(collectionIndexTransition);
            float_X const D = boundBoundTransitionDataBox.cxin4(collectionIndexTransition);
            float_X const a = boundBoundTransitionDataBox.cxin5(collectionIndexTransition);

            // calculate gaunt Factor
            float_X g = 0._X;

            // no need to check for U <= 0, since always U > 0,
            //  due to definition as central energy of bin of energy histogram

            // not a physical forbidden transition, electron has enough energy  higher than deltaE of transition
            if(U >= 1._X)
            {
                float_X const logU = math::log(U);

                if((A == 0._X) && (B == 0._X) && (C == 0._X) && (D == 0._X) && (a == 0._X))
                {
                    // use mewe approximation if all gaunt coefficients are zero
                    g = 0.15 + 0.28 * logU;
                }
                else
                {
                    auto const aPlusU = a + U;
                    // avoid division by 0
                    if(aPlusU != 0._X)
                        // chuung approximation
                        g = A * logU + B + C / aPlusU + D / (aPlusU * aPlusU); // unitless
                }

                bool const gauntFitUnPhysical = (g < 0._X);
                if(gauntFitUnPhysical)
                {
                    //  untiless
                    g = 0._X;
                }
            }

            // unitless
            return g;
        }

        //! check for NaNs and casting overflows in Ratio
        template<typename T_AtomicStateDataBox>
        HDINLINE static void debugChecksMultiplicity(
            float_X Ratio,
            uint32_t const lowerStateClctIdx,
            uint32_t const upperStateClctIdx,
            T_AtomicStateDataBox const atomicStateDataBox)
        {
            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;

            // Ratio not NaN
            if(!(Ratio == Ratio)) // only true if nan
                printf(
                    "atomicPhysics ERROR: NaN multiplicityConfigNumber ratio\n"
                    "   upperStateClctIdx %u ,lowerStateClctIdx %u ,(energy electron/energyDifference) %f\n",
                    upperStateClctIdx,
                    lowerStateClctIdx,
                    Ratio);

            // no overflow in float_X cast
            if(relativeTest(
                   atomicStateDataBox.multiplicity(lowerStateClctIdx)
                       / atomicStateDataBox.multiplicity(upperStateClctIdx),
                   float_64(float_X(
                       atomicStateDataBox.multiplicity(lowerStateClctIdx)
                       / atomicStateDataBox.multiplicity(upperStateClctIdx))),
                   1e-7))
                printf("atomicPhysics ERROR: overflow in multiplicityConfigNumber-ratio cast to float_X\n");
        }

    public:
        /** collisional cross section for a given bound-bound transition and electron energy
         *
         * @tparam T_AtomicStateDataBox }instantiated type of dataBox
         * @tparam T_BoundBoundTransitionDataBox instantiated type of dataBox
         * @tparam T_excitation true =^= excitation, false =^= deexcitation, direction of transition
         *
         * @param energyElectron kinetic energy of interacting free electron(/electron bin), [eV]
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 10^6*b = 10^(-22)m^2; (b == barn = 10^(-28) m^2)
         *
         * @attention assumes that atomicStateDataBox and boundBoundTransitionDataBox belong
         *      to the same AtomicData instance
         */
        template<typename T_AtomicStateDataBox, typename T_BoundBoundTransitionDataBox, bool T_excitation>
        HDINLINE static float_X collisionalBoundBoundCrossSection(
            float_X const energyElectron, // [eV]
            uint32_t const transitionCollectionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundBoundTransitionDataBox const boundBoundTransitionDataBox)
        {
            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;

            uint32_t const upperStateClctIdx
                = boundBoundTransitionDataBox.upperStateCollectionIndex(transitionCollectionIndex);
            uint32_t const lowerStateClctIdx
                = boundBoundTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);

            // eV
            float_X const energyDifference = picongpu::particles::atomicPhysics::DeltaEnergyTransition ::
                get<T_AtomicStateDataBox, T_BoundBoundTransitionDataBox>(
                    transitionCollectionIndex,
                    atomicStateDataBox,
                    boundBoundTransitionDataBox);

            if constexpr(picongpu::atomicPhysics::debug::rateCalculation::DEBUG_CHECKS)
                if(energyDifference < 0._X)
                {
                    printf("atomicPhysics ERROR: upper and lower state inverted in "
                           "collisionalBoundBoundCrossSection() call\n");
                    return 0._X;
                }

            // check whether electron has enough kinetic energy
            if constexpr(T_excitation)
            {
                // transition physical impossible, insufficient electron energy
                if(energyDifference > energyElectron)
                {
                    return 0._X;
                }
            }

            // unitless * unitless = unitless
            float_X const collisionalOscillatorStrength = static_cast<float_X>(
                boundBoundTransitionDataBox.collisionalOscillatorStrength(transitionCollectionIndex)); // unitless

            /* formula: scalingConstant * E_Ry^2/deltaE_Trans^2 * f * deltaE_Trans/E_kin * g
             *
             * E_Ry         ... Rydberg Energy
             * deltaE_Trans ... (energy upper state - energy lower state)
             * f            ... oscillator strength
             * E_kin        ... kinetic energy of interacting electron
             * g            ... gauntFactor
             */

            // (unitless * m)^2 / (unitless * m^2/1e6b) = m^2 / m^2 * 1e6b = 1e6b
            constexpr float_X scalingConstant = static_cast<float_X>(
                8. * pmacc::math::cPow(picongpu::PI * sim.si.getBohrRadius(), u8(2u))
                / (1.e-22)); // [1e6b], ~ 2211,01 * 1e6b
            // 1e6b
            constexpr float_X constantPart
                = scalingConstant * static_cast<float_X>(pmacc::math::cPow(sim.si.getRydbergEnergy(), u8(2u)));
            // [1e6b * (eV)^2]

            // 1e6b*(eV)^2 / (eV)^2 * unitless * (eV)/(eV) * unitless = 1e6b
            float_X crossSection_butGaunt = constantPart / math::sqrt(3._X)
                / pmacc::math::cPow(energyDifference, u8(2u)) * collisionalOscillatorStrength
                * (energyDifference / energyElectron);
            // [1e6b]

            float_X result = 0.0_X;

            // safeguard against negative cross sections due to imperfect approximations
            if(crossSection_butGaunt > 0._X)
            {
                // eV
                float_X U = energyElectron / energyDifference;
                // unitless
                // 1.0 is default value for excitation
                float_X ratio = 1._X;
                if constexpr(!T_excitation)
                {
                    // deexcitation
                    //      different multiplicityConfigNumber for deexcitation
                    // unitless
                    ratio = static_cast<float_X>(
                        atomicStateDataBox.multiplicity(lowerStateClctIdx)
                        / atomicStateDataBox.multiplicity(upperStateClctIdx));

                    if constexpr(picongpu::atomicPhysics::debug::rateCalculation::DEBUG_CHECKS)
                        debugChecksMultiplicity(ratio, lowerStateClctIdx, upperStateClctIdx, atomicStateDataBox);
                    // equal to U = (energyElectron + energyDifference) / energyDifference;
                    U += 1.0_X;
                }

                // [1e6b]
                result = crossSection_butGaunt * ratio
                    * gauntFactor(U, transitionCollectionIndex, boundBoundTransitionDataBox);
            }
            return result;
        }

        /** rate for collisional bound-bound transition of ion with free electron bin
         *
         * uses second order integration(bin middle)
         *
         * @todo implement higher order integrations, Brian Marre, 2022
         *
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundBoundTransitionDataBox instantiated type of dataBox
         * @tparam T_excitation true =^= excitation, false =^= deexcitation, direction of transition
         *
         * @param energyElectron kinetic energy of interacting electron(/electron bin), [eV]
         * @param energyElectronBinWidth energy width of electron bin, [eV]
         * @param densityElectrons [1/(m^3 * eV)]
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 1/sim.unit.time()
         */
        template<typename T_AtomicStateDataBox, typename T_BoundBoundTransitionDataBox, bool T_excitation>
        HDINLINE static float_X rateCollisionalBoundBoundTransition(
            float_X const energyElectron, // [eV]
            float_X const energyElectronBinWidth, // [eV]
            float_X const densityElectrons, // [1/(sim.unit.length()^3*eV)]
            uint32_t const transitionCollectionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundBoundTransitionDataBox const boundBoundTransitionDataBox)
        {
            float_X const sigma = collisionalBoundBoundCrossSection<
                T_AtomicStateDataBox,
                T_BoundBoundTransitionDataBox,
                T_excitation>(
                energyElectron, // [eV]
                transitionCollectionIndex,
                atomicStateDataBox,
                boundBoundTransitionDataBox); // [1e6*b]

            float_X const result = picongpu::particles2::atomicPhysics::rateCalculation::collisionalRate(
                energyElectron,
                energyElectronBinWidth,
                densityElectrons,
                sigma);

            if(result < 0._X)
            {
                printf(
                    "atomicPhysics ERROR: negative bound-bound collisional rate, "
                    "transition: %u\n",
                    transitionCollectionIndex);
            }

            return result;
        }

        /** rate of spontaneous photon emission for a given bound-bound transition
         *
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundBoundTransitionDataBox instantiated type of dataBox
         *
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 1/sim.unit.time(), usually Delta_T_SI ... PIC time step length
         */
        template<typename T_AtomicStateDataBox, typename T_BoundBoundTransitionDataBox>
        HDINLINE static float_X rateSpontaneousRadiativeDeexcitation(
            uint32_t const transitionCollectionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundBoundTransitionDataBox const boundBoundTransitionDataBox)
        {
            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;

            // short hands for constants in SI
            // m/s
            constexpr float_64 c_SI = picongpu::sim.si.getSpeedOfLight();
            // kg
            constexpr float_64 m_e_SI = picongpu::sim.si.getElectronMass();
            // C
            constexpr float_64 e_SI = picongpu::sim.si.getElectronCharge();

            // C/(Vm)
            constexpr float_64 mue0_SI = picongpu::sim.si.getMue0();
            // unitless
            constexpr float_64 pi = picongpu::PI;
            // Js
            constexpr float_64 hbar_SI = sim.si.getHbar();

            uint32_t const upperStateClctIdx
                = boundBoundTransitionDataBox.upperStateCollectionIndex(transitionCollectionIndex);
            uint32_t const lowerStateClctIdx
                = boundBoundTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);

            // eV
            float_X deltaEnergyTransition = static_cast<float_X>(
                atomicStateDataBox.energy(upperStateClctIdx) - atomicStateDataBox.energy(lowerStateClctIdx));

            // J/(eV) / (Js) * s/sim.unit.time() = J/J * s/s * 1/(eV * sim.unit.time())
            constexpr float_X scalingConstantPhotonFrequency
                = static_cast<float_X>(sim.si.conv().eV2Joule(1.0) / (2 * pi * hbar_SI) * picongpu::sim.unit.time());

            /// @attention actual SI frequency, NOT angular frequency
            // 1/sim.unit.time()
            float_X frequencyPhoton = deltaEnergyTransition * scalingConstantPhotonFrequency;

            // unitless
            float_X ratio = static_cast<float_X>(
                atomicStateDataBox.multiplicity(lowerStateClctIdx)
                / atomicStateDataBox.multiplicity(upperStateClctIdx));

            if constexpr(picongpu::atomicPhysics::debug::rateCalculation::DEBUG_CHECKS)
                debugChecksMultiplicity(ratio, lowerStateClctIdx, upperStateClctIdx, atomicStateDataBox);

            // (2 * pi * e^2)/(eps_0 * m_e * c^3 * s/sim.unit.time()) = (2 * pi * e^2 * mue_0) / (m_e * c *
            // s/sim.unit.time())
            /* (N/A^2 * (As)^2) / (kg * m/s * s/sim.unit.time()) = (A^2/A^2 *s^2 * N * sim.unit.time()) / (kg * m *
             * s/s) = (s^2 * kg*m/(s^2) * sim.unit.time()) / ( kg * m) = s^2/(s^2) (kg*m)/(kg*m) * sim.unit.time() =
             * sim.unit.time()
             */
            // sim.unit.time()
            constexpr float_X scalingConstantRate = static_cast<float_X>(
                (2. * pi * e_SI * e_SI * mue0_SI) / (m_e_SI * c_SI * picongpu::sim.unit.time()));

            /* [(2 * pi * e^2)/(eps_0 * m_e * c^3)] * nu^2 * g_new/g_old * faax
             * taken from https://en.wikipedia.org/wiki/Einstein_coefficients
             * s * (1/s)^2 = 1/s
             */
            // sim.unit.time() * 1/(sim.unit.time()^2) * unitless * unitless = 1/sim.unit.time()
            // 1/sim.unit.time()
            return scalingConstantRate * frequencyPhoton * frequencyPhoton * ratio
                * boundBoundTransitionDataBox.absorptionOscillatorStrength(transitionCollectionIndex);
        }

        /// @todo radiativeBoundBoundCrossSection, Brian Marre, 2022
        /// @todo rateRadiativeBoundBoundTransition, Brian Marre, 2022
    };
} // namespace picongpu::particles::atomicPhysics::rateCalculation
