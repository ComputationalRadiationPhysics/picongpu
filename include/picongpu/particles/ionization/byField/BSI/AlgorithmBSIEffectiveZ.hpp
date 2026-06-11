/*
 * SPDX-FileCopyrightText: Marco Garten, Jakob Trojok
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/IonizerReturn.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"
#include "picongpu/particles/traits/GetEffectiveNuclearCharge.hpp"
#include "picongpu/particles/traits/GetIonizationEnergies.hpp"
#include "picongpu/traits/attribute/GetChargeState.hpp"

/** @file AlgorithmBSIEffectiveZ.hpp
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
            struct AlgorithmBSIEffectiveZ
            {
                /** Functor implementation
                 *
                 * @tparam EType type of electric field
                 * @tparam ParticleType type of particle to be ionized
                 *
                 * @param eField electric field value at t=0
                 * @param parentIon particle instance to be ionized with position at t=0 and momentum at t=-1/2
                 *
                 * and "t" being with respect to the current time step (on step/half a step backward/-""-forward)
                 *
                 * @return ionization energy and number of new macro electrons to be created
                 * (current implementation supports only 0 or 1 per execution)
                 */
                template<typename EType, typename ParticleType>
                HDINLINE IonizerReturn operator()(EType const eField, ParticleType& parentIon)
                {
                    float_X const protonNumber
                        = picongpu::traits::GetAtomicNumbers<ParticleType>::type::numberOfProtons;
                    float_X chargeState = picongpu::traits::attribute::getChargeState(parentIon);

                    /* verify that ion is not completely ionized */
                    if(chargeState < protonNumber)
                    {
                        uint32_t cs = pmacc::math::float2int_rd(chargeState);
                        /* ionization potential in atomic units */
                        float_X const iEnergy =
                            typename picongpu::traits::GetIonizationEnergies<ParticleType>::type{}[cs];
                        float_X const ZEff =
                            typename picongpu::traits::GetEffectiveNuclearCharge<ParticleType>::type{}[cs];
                        /* critical field strength in atomic units */
                        float_X critField = iEnergy * iEnergy / (float_X(4.0) * ZEff);

                        /* ionization condition */
                        if(sim.pic.conv().eField2auEField(pmacc::math::l2norm(eField)) >= critField)
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
