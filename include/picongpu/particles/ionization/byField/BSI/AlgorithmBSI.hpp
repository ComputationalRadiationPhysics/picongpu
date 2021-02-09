/* Copyright 2014-2021 Marco Garten, Jakob Trojok
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
#include "picongpu/particles/traits/GetIonizationEnergies.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"
#include "picongpu/traits/attribute/GetChargeState.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/IonizerReturn.hpp"

/** @file AlgorithmBSI.hpp
 *
 * IONIZATION ALGORITHM for the BSI model
 *
 * - implements the calculation of ionization probability and changes charge states
 *   by decreasing the number of bound electrons
 * - is called with the IONIZATION MODEL, specifically by setting the flag in @see speciesDefinition.param
 */

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** Calculation for the Barrier Suppression Ionization model
             */
            struct AlgorithmBSI
            {
                /** Functor implementation
                 *
                 * \tparam EType type of electric field
                 * \tparam ParticleType type of particle to be ionized
                 *
                 * \param eField electric field value at t=0
                 * \param parentIon particle instance to be ionized with position at t=0 and momentum at t=-1/2
                 *
                 * and "t" being with respect to the current time step (on step/half a step backward/-""-forward)
                 *
                 * \return ionization energy and number of new macro electrons to be created
                 * (current implementation supports only 0 or 1 per execution)
                 */
                template<typename EType, typename ParticleType>
                HDINLINE IonizerReturn operator()(const EType eField, ParticleType& parentIon)
                {
                    float_X const protonNumber = GetAtomicNumbers<ParticleType>::type::numberOfProtons;
                    float_X const chargeState = attribute::getChargeState(parentIon);

                    /* verify that ion is not completely ionized */
                    if(chargeState < protonNumber)
                    {
                        uint32_t const cs = pmacc::math::float2int_rd(chargeState);
                        /* ionization potential in atomic units */
                        float_X const iEnergy = typename GetIonizationEnergies<ParticleType>::type{}[cs];
                        /* the charge that attracts the electron that is to be ionized:
                         * equals `protonNumber - no. allInnerElectrons`
                         */
                        float_X const effectiveCharge = chargeState + float_X(1.0);
                        /* critical field strength in atomic units */
                        float_X const critField = iEnergy * iEnergy / (float_X(4.0) * effectiveCharge);
                        /* ionization condition */
                        if(math::abs(eField) / ATOMIC_UNIT_EFIELD >= critField)
                        {
                            /* return ionization energy and number of macro electrons to produce */
                            return IonizerReturn{iEnergy, 1u};
                        }
                    }
                    return IonizerReturn{0.0, 0u};
                }
            };

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
