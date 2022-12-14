/* Copyright 2017-2021 Axel Huebl, Sudhir Sharma, Brian Marre
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

#include "picongpu/simulation_defines.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cstdint>

/** rate calculation from given atomic data, extracted from flylite, based on FLYCHK
 *
 *
 * References:
 * - Axel Huebl
 *  flylite, not yet published
 *
 *  - R. Mewe.
 *  "Interpolation formulae for the electron impact excitation of ions in
 *  the H-, He-, Li-, and Ne-sequences."
 *  Astronomy and Astrophysics 20, 215 (1972)
 *
 *  - H.-K. Chung, R.W. Lee, M.H. Chen.
 *  "A fast method to generate collisional excitation cross-sections of
 *  highly charged ions in a hot dense matter"
 *  High Energy Dennsity Physics 3, 342-352 (2007)
 */
namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            /** functor class containing calculation formulas of rates and cross sections
             *
             * @tparam T_AtomicDataBox ... type of atomic data box used, stores actual basic
             *      atomic input data
             * TODO: T_TypeIndex is accessible from T_ConfigNumber remove direct mention
             * @tparam T_TypeIndex ... data type of atomic state index used in configNumber, unitless
             * @tparam T_ConfigNumber ... type of storage object of atomic state
             * @tparam T_numLevels ... number of atomic levels modelled in configNumber, unitless
             * BEWARE: atomic data box input data is assumed to be in eV
             */
            template<typename T_AtomicDataBox, typename T_TypeIndex, typename T_ConfigNumber, uint8_t T_numLevels>
            class AtomicRate
            {
            public:
                // shorthands
                using Idx = T_TypeIndex;
                using AtomicDataBox = T_AtomicDataBox;
                using ConfigNumber = T_ConfigNumber;

                // data type of occupation number vector
                using LevelVector = pmacc::math::Vector<uint8_t,
                                                        T_numLevels>; // unitless

            public:
                /** binomial coefficient calculated using partial pascal triangle
                 *
        t         *      should be no problem in flychk data since largest value ~10^10
                 *      will become problem if all possible states are used
                 *
                 * TODO: add description of iteration,
                 * - algorithm tested against scipy.specia.binomial
                 *
                 * Source: https://www.tutorialspoint.com/binomial-coefficient-in-cplusplus;
                 *  22.11.2019
                 */
                DINLINE static float_64 binomialCoefficients(uint8_t n, uint8_t k)
                {
                    // check for limits, no check for < 0 necessary, since uint
                    if(n < k)
                    {
                        printf("invalid call binomial(n,k), with n < k");
                        return 0.f;
                    }

                    // reduce necessary steps using symmetry in k
                    if(k > n / 2._X)
                    {
                        k = n - k;
                    }

                    float_64 result = 1u;

                    for(uint8_t i = 1u; i <= k; i++)
                    {
                        result *= (n - i + 1) / static_cast<float_64>(i);
                    }
                    return result;
                }

                // number of different atomic configurations in an atomic state
                // @param idx ... index of atomic state, unitless
                // return unit: unitless
                template<typename T_Worker>
                DINLINE static float_64 Multiplicity(T_Worker const& worker, Idx configNumber)
                {
                    LevelVector const levelVector = ConfigNumber::getLevelVector(configNumber); // unitless

                    float_64 result = 1u;

                    // @TODO: chnage power function call to explicit multipication, BrianMarre 2020
                    for(uint8_t i = 0u; i < T_numLevels; i++)
                    {
                        result *= binomialCoefficients(
                            static_cast<uint8_t>(2u)
                                * pmacc::math::cPow(i + static_cast<uint8_t>(1u), static_cast<uint8_t>(2u)),
                            levelVector[i]); // unitless
                    }

                    return result; // unitless
                }

                /** gaunt factor like suppression of crosssection
                 *
                 * @param energyDifference ... difference of energy between atomic states, unit: ATOMIC_UNIT_ENERGY
                 * @param energyElectron ... energy of electron, unit ATOMIC_UNIT_ENERGY
                 * @param indexTransition ... internal index of transition in atomicDataBox
                 *      use findIndexTransition method of atomicDataBox and screen for not found value
                 *      BEWARE: method assumes that indexTransition is valid, undefined behaviour otherwise
                 *
                 * return unit: unitless
                 */
                DINLINE static float_X gauntFactor(
                    float_X energyDifference, // unit: ATOMIC_UNIT_ENERGY
                    float_X energyElectron, // unit: ATOMIC_UNIT_ENERGY
                    uint32_t indexTransition, // unitless
                    AtomicDataBox atomicDataBox)
                {
                    // get gaunt coeficients, unit: unitless
                    float_X const A = atomicDataBox.getCinx1(indexTransition);
                    float_X const B = atomicDataBox.getCinx2(indexTransition);
                    float_X const C = atomicDataBox.getCinx3(indexTransition);
                    float_X const D = atomicDataBox.getCinx4(indexTransition);
                    float_X const a = atomicDataBox.getCinx5(indexTransition);

                    // calculate gaunt Factor
                    float_X const U = energyElectron / energyDifference; // unit: unitless
                    float_X const g = A * math::log(U) + B + C / (U + a) + D / ((U + a) * (U + a)); // unitless

                    return g * (U > 1.0); // unitless
                }

            public:
                // return unit: ATOMIC_UNIT_ENERGY
                template<typename T_Worker>
                DINLINE static float_X energyDifference(
                    T_Worker const& worker,
                    Idx const oldConfigNumber, // unitless
                    Idx const newConfigNumber, // unitless
                    AtomicDataBox atomicDataBox)
                {
                    return (atomicDataBox(newConfigNumber) - atomicDataBox(oldConfigNumber));
                    // unit: ATOMIC_UNIT_ENERGY
                }

                /** returns the cross section for collisional exitation and deexcitation
                 *
                 * NOTE: does not check whether electron has enough energy, this is
                 * expected to be done by caller
                 *
                 * @param energyElectron ... kinetic energy only, unit: ATOMIC_UNIT_ENERGY
                 * return unit: m^2
                 */
                template<typename T_Worker>
                DINLINE static float_X collisionalExcitationCrosssection(
                    T_Worker const& worker,
                    Idx const oldConfigNumber, // unitless
                    Idx const newConfigNumber, // unitless
                    uint32_t const transitionIndex, // unitless
                    float_X energyElectron, // unit: ATOMIC_UNIT_ENERGY
                    AtomicDataBox const atomicDataBox)
                {
                    // energy difference between atomic states
                    float_X m_energyDifference = energyDifference(
                        worker,
                        oldConfigNumber,
                        newConfigNumber,
                        atomicDataBox); // unit: ATOMIC_UNIT_ENERGY

                    // case no physical transition possible, since insufficient electron energy
                    if(m_energyDifference > energyElectron)
                        return 0._X;

                    float_X Ratio; // unitless

                    // excitation or deexcitation
                    if(m_energyDifference < 0._X)
                    {
                        // deexcitation
                        m_energyDifference = -m_energyDifference; // unit: ATOMIC_UNIT_ENERGY

                        // ratio due to multiplicity
                        // unitless/unitless * (AU + AU) / AU = unitless
                        Ratio = static_cast<float_X>(
                                    (Multiplicity(worker, newConfigNumber)) / (Multiplicity(worker, oldConfigNumber)))
                            * (energyElectron + m_energyDifference) / energyElectron; // unitless

                        // security check for NaNs in Ratio and debug outputif present
                        if(!(Ratio == Ratio)) // only true if nan
                        {
                            printf(
                                "Warning: NaN in ratio calculation, ask developer for more information\n"
                                "   newIdx %u ,oldIdx %u ,energyElectron_SI %f ,energyDifference_m %f",
                                static_cast<uint32_t>(newConfigNumber),
                                static_cast<uint32_t>(oldConfigNumber),
                                energyElectron,
                                m_energyDifference);
                        }

                        energyElectron = energyElectron + m_energyDifference; // unit; ATOMIC_UNIT_ENERGY
                    }
                    else
                    {
                        // excitation
                        Ratio = 1._X; // unitless
                    }

                    // unitless * unitless = unitless
                    float_X const collisionalOscillatorStrength
                        = Ratio * atomicDataBox.getCollisionalOscillatorStrength(transitionIndex); // unitless

                    // physical constants
                    // (unitless * m)^2 / unitless = m^2
                    float_X c0_SI = float_X(
                        8._X
                        * pmacc::math::cPow(
                            picongpu::PI * picongpu::SI::BOHR_RADIUS,
                            static_cast<uint8_t>(2u))
                        / math::sqrt(3._X)); // uint: m^2, SI

                    // scaling constants * E_Ry^2/deltaE_Trans^2 * f * deltaE_Trans/E_kin
                    // m^2 * (AUE/AUE)^2 * unitless * AUE/AUE * unitless<-[ J, J, unitless, unitless ] = m^2
                    // AUE =^= ATOMIC_UNIT_ENERGY
                    float_X crossSection_SI = c0_SI * (1._X / 4._X) / (m_energyDifference * m_energyDifference)
                        * collisionalOscillatorStrength * (m_energyDifference / energyElectron)
                        * gauntFactor(m_energyDifference,
                                      energyElectron,
                                      transitionIndex,
                                      atomicDataBox); // unit: m^2, SI

                    // safeguard against negative cross sections due to imperfect approximations
                    if(crossSection_SI < 0._X)
                    {
                        return 0._X;
                    }

                    return crossSection_SI;
                }

                /** returns total cross section for a specific electron energy
                 *
                 * @param energyElectron ... kinetic energy only, unit: ATOMIC_UNIT_ENERGY
                 * return unit: m^2, SI
                 */
                template<typename T_Worker>
                DINLINE static float_X totalElectronInteractionCrossSection(
                    T_Worker const& worker,
                    float_X const energyElectron, // unit: ATOMIC_UNIT_ENERGY
                    AtomicDataBox const atomicDataBox,
                    bool const debug = false)
                {
                    float_X crossSection = 0._X; // unit: m^2, SI

                    Idx lowerStateConfigNumber;
                    Idx upperStateConfigNumber;
                    uint32_t numberTransitions;
                    uint32_t startIndexBlock;
                    uint32_t transitionIndex;

                    // iterate over all transitions
                    for(uint32_t i = 0u; i < atomicDataBox.getNumStates(); i++)
                    {
                        lowerStateConfigNumber = atomicDataBox.getAtomicStateConfigNumberIndex(i);
                        numberTransitions = atomicDataBox.getNumberTransitions(i);
                        startIndexBlock = atomicDataBox.getStartIndexBlock(i);

                        for(uint32_t j = 0u; j < numberTransitions; j++)
                        {
                            upperStateConfigNumber = atomicDataBox.getUpperConfigNumberTransition(startIndexBlock + j);
                            transitionIndex = startIndexBlock + j;

                            // check excitation possible with electron energy
                            if(AtomicRate::energyDifference(
                                   worker,
                                   lowerStateConfigNumber,
                                   upperStateConfigNumber,
                                   atomicDataBox)
                               <= energyElectron)
                            {
                                // excitation cross section
                                crossSection += collisionalExcitationCrosssection(
                                    worker,
                                    lowerStateConfigNumber, // unitless
                                    upperStateConfigNumber, // unitless
                                    transitionIndex, // unitless
                                    energyElectron, // unit: ATOMIC_UNIT_ENERGY
                                    atomicDataBox); // unit: m^2, SI
                            }


                            // NaN/inf check
                            if(crossSection == crossSection + 1)
                            {
                                printf(
                                    "crossSectionExcitation %f lowerStateConfigNumber %u, "
                                    "upperStateConfigNumber %u energyElectron %f \n",
                                    collisionalExcitationCrosssection(
                                        worker,
                                        lowerStateConfigNumber, // unitless
                                        upperStateConfigNumber, // unitless
                                        transitionIndex, // uintless
                                        energyElectron, // unit: ATOMIC_UNIT_ENERGY
                                        atomicDataBox),
                                    lowerStateConfigNumber,
                                    upperStateConfigNumber,
                                    energyElectron);
                            }

                            // deexcitation crosssection, always possible
                            crossSection += collisionalExcitationCrosssection(
                                worker,
                                upperStateConfigNumber,
                                lowerStateConfigNumber,
                                transitionIndex,
                                energyElectron,
                                atomicDataBox); // unit: m^2, SI

                            // NaN/inf check
                            if(crossSection == crossSection + 1)
                            {
                                printf(
                                    " crossSectionDeExcitation %f \n",
                                    collisionalExcitationCrosssection(
                                        worker,
                                        upperStateConfigNumber, // unitless
                                        lowerStateConfigNumber, // unitless
                                        transitionIndex, // uintless
                                        energyElectron, // unit: ATOMIC_UNIT_ENERGY
                                        atomicDataBox));
                            }
                        }
                    }

                    return crossSection; // unit: m^2, SI
                }

                // return unit: 1/s, SI
                /** rate function for interaction of ion with free electron
                 * uses 1th order integration <-> a = 0, => T_minOrderApprox = 1
                 * TODO: implement higher order integration
                 * TODO: change density to internal units
                 * TODO: change return unit to internal units
                 *
                 * @param energyElectron ... kinetic energy only, unit: ATOMIC_UNIT_ENERGY
                 * @param energyElectronBinWidth ... unit: ATOMIC_UNIT_ENERGY
                 * @param densityElectrons ... unit: 1/(m^3 * J)
                 * @param atomicDataBox ... acess to input atomic data
                 *
                 * return unit: 1/s ... SI
                 */
                template<typename T_Worker>
                DINLINE static float_X RateFreeElectronInteraction(
                    T_Worker const& acc,
                    Idx const oldState, // unit: unitless
                    Idx const newState, // unit: unitless
                    uint32_t const transitionIndex,
                    float_X const energyElectron, // unit: ATOMIC_UNIT_ENERGY
                    float_X const energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                    float_X const densityElectrons, // unit: 1/(m^3*ATOMIC_UNIT_ENERGY)
                    AtomicDataBox const atomicDataBox)
                {
                    // Notation Note: AU is a shorthand for ATOMIC_UNIT_ENERGY in this context

                    // constants in SI
                    constexpr float_64 c_SI = picongpu::SI::SPEED_OF_LIGHT_SI; // unit: m/s, SI
                    constexpr float_64 m_e_SI = picongpu::SI::ELECTRON_MASS_SI; // unit: kg, SI

                    const float_64 E_e_SI = energyElectron * picongpu::UNITCONV_AU_to_eV * UNITCONV_eV_to_Joule;
                    // unit: J, SI

                    const float_X sigma_SI = collisionalExcitationCrosssection(
                        acc,
                        oldState, // unitless
                        newState, // unitless
                        transitionIndex,
                        energyElectron, // unit: ATOMIC_UNIT_ENERGY
                        atomicDataBox); // unit: (m^2), SI

                    // AU * m^2 * 1/(m^3*AU) * m/s * sqrt( unitless - [ ( (kg*m^2/s^2)/J )^2 = Nm/J = J/J = unitless ]
                    // ) = AU/AU m^3/m^3 * 1/s
                    return energyElectronBinWidth * sigma_SI * densityElectrons * c_SI
                        * math::sqrt(
                               1.0
                               - 1.0
                                   / ((1._X + E_e_SI / (m_e_SI * c_SI * c_SI))
                                      * (1._X + E_e_SI / (m_e_SI * c_SI * c_SI))));
                    // unit: 1/s; SI
                }

                // return unit: 1/s, SI
                /** rate function for spontaneous photon emission
                 * TODO: change return unit to internal units
                 *
                 * @param atomicDataBox ... acess to input atomic data
                 *
                 * return unit: 1/s ... SI
                 */
                template<typename T_Worker>
                DINLINE static float_X RateSpontaneousPhotonEmission(
                    T_Worker const& acc,
                    Idx const oldState, // unit: unitless
                    Idx const newState, // unit: unitless
                    uint32_t const transitionIndex,
                    float_X deltaEnergyTransition, // unit: ATOMIC_UNIT_ENERGY
                    AtomicDataBox const atomicDataBox)
                {
                    // Notation Note: AU is a shorthand for ATOMIC_UNIT_ENERGY in this context

                    // constants in SI
                    constexpr float_64 c_SI = picongpu::SI::SPEED_OF_LIGHT_SI; // unit: m/s, SI
                    constexpr float_64 m_e_SI = picongpu::SI::ELECTRON_MASS_SI; // unit: kg, SI
                    constexpr float_64 e_SI = picongpu::SI::ELECTRON_CHARGE_SI; // unit: C, SI

                    constexpr float_64 mue0_SI = picongpu::SI::MUE0_SI; // unit: C/(Vm), SI
                    constexpr float_64 pi = picongpu::PI; // unit: unitless
                    constexpr float_64 hbar_SI = picongpu::SI::HBAR_SI; // unit: Js, SI

                    constexpr float_64 au_SI = picongpu::SI::ATOMIC_UNIT_ENERGY; // unit: J, SI

                    // (2 * pi * e^2)/(eps_0 * m_e * c^3) = (2 * pi * e^2 * mue_0) / (m_e * c)
                    constexpr float_X constFactor
                        = static_cast<float_X>((2 * pi * e_SI * e_SI * mue0_SI) / (m_e_SI * c_SI));
                    // unit: ((As)^2 * N/A^2) /(kg * m/s) = A^2/A^2 *s^2 * N / ( kg * m/s )
                    // = s^2 * kg*m/s^2 / ( kg * m/s ) = 1/(1/s) = s, SI

                    float_X frequencyPhoton = static_cast<float_X>(
                        (static_cast<float_64>(deltaEnergyTransition) * au_SI) / hbar_SI); // unit: 1/s
                    // NOTE: this is an actual frequency not an angular frequency

                    float_X Ratio = static_cast<float_X>(Multiplicity(acc, newState) / Multiplicity(acc, oldState));
                    // unit: unitless

                    // (2 * pi * e^2)/(eps_0 * m_e * c^3) * nu^2 * g_new/g_old * faax
                    // taken from https://en.wikipedia.org/wiki/Einstein_coefficients
                    // s * (1/s)^2 = 1/s
                    return constFactor * frequencyPhoton * frequencyPhoton * Ratio
                        * atomicDataBox.getAbsorptionOscillatorStrength(transitionIndex);
                    // unit: 1/s; SI
                }

                /** returns the total rate of all transitions from the given state
                 * to any other state, using any process
                 *
                 * return unit: 1/s, SI
                 */
                template<typename T_Worker>
                DINLINE static float_X totalRate(
                    T_Worker const& worker,
                    Idx oldState, // unitless
                    float_X energyElectron, // unit: ATOMIC_UNIT_ENERGY
                    float_X energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                    float_X densityElectrons, // unit: 1/(m^3*ATOMIC_UNIT_ENERGY)
                    AtomicDataBox atomicDataBox) // unit: 1/s, SI
                {
                    // NOTE on Notation: the term upper-/lowerState refers
                    //  to the energy of the state
                    float_X totalRate = 0._X; // unit: 1/s, SI

                    Idx lowerState;
                    Idx upperState;
                    uint32_t startIndexBlock;
                    uint32_t indexTransition;

                    for(uint32_t i = 0u; i < atomicDataBox.getNumStates(); i++)
                    {
                        lowerState = atomicDataBox.getAtomicStateConfigNumberIndex(i);
                        startIndexBlock = atomicDataBox.getStartIndexBlock(i);

                        for(uint32_t j = 0u; j < atomicDataBox.getNumberTransitions(i); j++)
                        {
                            indexTransition = startIndexBlock + j;
                            upperState = atomicDataBox.getUpperConfigNumberTransition(indexTransition);

                            // transitions with oldState as upper state
                            if(upperState == oldState)
                            {
                                totalRate += RateFreeElectronInteraction(
                                    worker,
                                    oldState, // unitless
                                    lowerState, // newstate, unitless
                                    indexTransition,
                                    energyElectron, // unit: ATOMIC_UNIT_ENERGY
                                    energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                                    densityElectrons, // unit: 1/(m^3*ATOMIC_UNIT_ENERGY)
                                    atomicDataBox); // unit: 1/s, SI

                                float_X deltaEnergyTransition = energyDifference(
                                    worker,
                                    oldState,
                                    lowerState,
                                    atomicDataBox); // unit: ATOMIC_UNIT_ENERGY

                                totalRate += RateSpontaneousPhotonEmission(
                                    worker,
                                    oldState,
                                    lowerState,
                                    indexTransition,
                                    deltaEnergyTransition, // unit: ATOMIC_UNIT_ENERGY
                                    atomicDataBox); // unit: 1/s, SI
                            }

                            // transitions with oldState as lower State
                            if((lowerState == oldState)
                               && (AtomicRate::energyDifference(worker, lowerState, upperState, atomicDataBox)
                                   <= energyElectron))
                                totalRate += RateFreeElectronInteraction(
                                    worker,
                                    oldState, // unitless
                                    upperState, // newstate, unitless
                                    indexTransition,
                                    energyElectron, // unit: ATOMIC_UNIT_ENERGY
                                    energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                                    densityElectrons, // unit: 1/(m^3*ATOMIC_UNIT_ENERGY)
                                    atomicDataBox); // unit: 1/s, SI

                            // else do nothing
                        }
                    }

                    return totalRate; // unit: 1/s, SI
                }

                /** returns the total rate of all spontaneous transitions
                 * from the given state to any other state
                 *
                 * return unit: 1/s, SI
                 */
                template<typename T_Worker>
                DINLINE static float_X totalSpontaneousRate(
                    T_Worker const& worker,
                    Idx oldState, // unitless
                    AtomicDataBox atomicDataBox) // unit: 1/s, SI
                {
                    // NOTE on Notation: the term upper-/lowerState refers
                    //  to the energy of the state
                    float_X totalRate = 0._X; // unit: 1/s, SI

                    Idx lowerState;
                    Idx upperState;
                    uint32_t startIndexBlock;
                    uint32_t indexTransition;

                    for(uint32_t i = 0u; i < atomicDataBox.getNumStates(); i++)
                    {
                        lowerState = atomicDataBox.getAtomicStateConfigNumberIndex(i);
                        startIndexBlock = atomicDataBox.getStartIndexBlock(i);

                        for(uint32_t j = 0u; j < atomicDataBox.getNumberTransitions(i); j++)
                        {
                            indexTransition = startIndexBlock + j;
                            upperState = atomicDataBox.getUpperConfigNumberTransition(indexTransition);

                            // transitions with oldState as upper state
                            if(upperState == oldState)
                            {
                                float_X deltaEnergyTransition = energyDifference(
                                    worker,
                                    oldState,
                                    lowerState,
                                    atomicDataBox); // unit: ATOMIC_UNIT_ENERGY

                                totalRate += RateSpontaneousPhotonEmission(
                                    worker,
                                    oldState,
                                    lowerState,
                                    indexTransition,
                                    deltaEnergyTransition, // unit: ATOMIC_UNIT_ENERGY
                                    atomicDataBox); // unit: 1/s, SI
                            }

                            // transitions with oldState as lower State
                            // no contribution

                            // else do nothing
                        }
                    }

                    return totalRate; // unit: 1/s, SI
                }
            };

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
