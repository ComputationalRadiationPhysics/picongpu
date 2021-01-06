/* Copyright 2015-2021 Marco Garten, Jakob Trojok
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

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"
#include "picongpu/particles/traits/GetIonizationEnergies.hpp"
#include "picongpu/traits/attribute/GetChargeState.hpp"
#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/algorithms/math/floatMath/floatingPoint.tpp>
#include "picongpu/particles/ionization/utilities.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/IonizerReturn.hpp"

/** \file AlgorithmADK.hpp
 *
 * IONIZATION ALGORITHM for the ADK model
 *
 * - implements the calculation of ionization probability and changes charge
 *   states by decreasing the number of bound electrons
 * - is called with the IONIZATION MODEL, specifically by setting the flag in
 *   speciesDefinition.param
 */


namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** Calculation for the Ammosov-Delone-Krainov tunneling model
             *
             * for either linear or circular laser polarization
             *
             * \tparam T_linPol boolean value that is true for lin. pol. and false for circ. pol.
             */
            template<bool T_linPol>
            struct AlgorithmADK
            {
                /** Functor implementation
                 * \tparam EType type of electric field
                 * \tparam BType type of magnetic field
                 * \tparam ParticleType type of particle to be ionized
                 *
                 * \param bField magnetic field value at t=0
                 * \param eField electric field value at t=0
                 * \param parentIon particle instance to be ionized with position at t=0 and momentum at t=-1/2
                 * \param randNr random number, equally distributed in range [0.:1.0]
                 *
                 * \return ionization energy and number of new macro electrons to be created
                 */
                template<typename EType, typename BType, typename ParticleType>
                HDINLINE IonizerReturn
                operator()(const BType bField, const EType eField, ParticleType& parentIon, float_X randNr)
                {
                    float_X const protonNumber = GetAtomicNumbers<ParticleType>::type::numberOfProtons;
                    float_X const chargeState = attribute::getChargeState(parentIon);

                    /* verify that ion is not completely ionized */
                    if(chargeState < protonNumber)
                    {
                        uint32_t const cs = pmacc::math::float2int_rd(chargeState);
                        float_X const iEnergy = typename GetIonizationEnergies<ParticleType>::type{}[cs];

                        constexpr float_X pi = pmacc::math::Pi<float_X>::value;
                        /* electric field in atomic units - only absolute value */
                        float_X const eInAU = math::abs(eField) / ATOMIC_UNIT_EFIELD;

                        /* the charge that attracts the electron that is to be ionized:
                         * equals `protonNumber - #allInnerElectrons`
                         */
                        float_X const effectiveCharge = chargeState + float_X(1.0);
                        /* effective principal quantum number (unitless) */
                        float_X const nEff = effectiveCharge / math::sqrt(float_X(2.0) * iEnergy);
                        /* nameless variable for convenience dFromADK*/
                        float_X const dBase = float_X(4.0) * util::cube(effectiveCharge) / (eInAU * util::quad(nEff));
                        float_X const dFromADK = math::pow(dBase, nEff);

                        /* ionization rate (for CIRCULAR polarization)*/
                        float_X rateADK = eInAU * util::square(dFromADK) / (float_X(8.0) * pi * effectiveCharge)
                            * math::exp(float_X(-2.0) * util::cube(effectiveCharge)
                                        / (float_X(3.0) * util::cube(nEff) * eInAU));

                        /* in case of linear polarization the rate is modified by an additional factor */
                        if(T_linPol)
                        {
                            /* factor from averaging over one laser cycle with LINEAR polarization */
                            float_X const polarizationFactor = math::sqrt(
                                float_X(3.0) * util::cube(nEff) * eInAU / (pi * util::cube(effectiveCharge)));

                            rateADK *= polarizationFactor;
                        }

                        /* simulation time step in atomic units */
                        float_X const timeStepAU = float_X(DELTA_T / ATOMIC_UNIT_TIME);
                        /* ionization probability
                         *
                         * probability = rate * time step
                         * --> for infinitesimal time steps
                         *
                         * the whole ensemble should then follow
                         * P = 1 - exp(-rate * time step) if the laser wavelength is
                         * sampled well enough
                         */
                        float_X const probADK = rateADK * timeStepAU;

                        /* ionization condition */
                        if(randNr < probADK)
                        {
                            /* return ionization energy and number of macro electrons to produce */
                            return IonizerReturn{iEnergy, 1u};
                        }
                    }
                    /* no ionization */
                    return IonizerReturn{0.0, 0u};
                }
            };

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
