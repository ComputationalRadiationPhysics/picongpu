/**
 * Copyright 2015 Marco Garten
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

#include "types.h"
#include "simulation_defines.hpp"
#include "particles/traits/GetAtomicNumbers.hpp"
#include "particles/traits/GetIonizationEnergies.hpp"
#include "traits/attribute/GetChargeState.hpp"
#include "algorithms/math/floatMath/floatingPoint.tpp"
#include "particles/ionization/utilities.hpp"

/** \file AlgorithmADK.hpp
 *
 * IONIZATION ALGORITHM for the ADK model
 *
 * - implements the calculation of ionization probability and changes charge states
 *   by decreasing the number of bound electrons
 * - is called with the IONIZATION MODEL, specifically by setting the flag in @see speciesDefinition.param */

namespace picongpu
{
namespace particles
{
namespace ionization
{

    /** \struct AlgorithmADK
     *
     * \brief calculation for the Ammosov-Delone-Krainov tunneling model */
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
         */
        template<typename EType, typename BType, typename ParticleType >
        HDINLINE void
        operator()( const BType bField, const EType eField, ParticleType& parentIon, float_X randNr )
        {

            const float_X protonNumber  = GetAtomicNumbers<ParticleType>::type::numberOfProtons;
            float_X chargeState         = attribute::getChargeState(parentIon);
            uint32_t cs                 = math::float2int_rd(chargeState);
            const float_X iEnergy       = GetIonizationEnergies<ParticleType>::type()[cs];

            const float_X pi    = precisionCast<float_X>(M_PI);
            /* electric field in atomic units - only absolute value */
            float_X eInAU       = math::abs(eField) / ATOMIC_UNIT_EFIELD;

            /* effective principal quantum number (unitless) */
            float_X nEff        = protonNumber / math::sqrt(float_X(2.0) * iEnergy );
            /* nameless variable for convenience dFromADK*/
            float_X dBase       = float_X(4.0) * util::cube(protonNumber) / ( eInAU * util::quad(nEff)) ;
            float_X dFromADK    = math::pow(dBase,nEff);

            /* ionization rate */
            float_X rateADK     = math::sqrt(float_X(3.0) * util::cube(nEff) * eInAU / (pi * util::cube(protonNumber))) \
                                    * eInAU * util::square(dFromADK) / (float_X(8.0) * pi * protonNumber) \
                                    * math::exp(float_X(-2.0) * util::cube(protonNumber) / (float_X(3.0) * util::cube(nEff) * eInAU));

            /* simulation time step in atomic units */
            const float_X timeStepAU = float_X(DELTA_T / ATOMIC_UNIT_TIME);
            /* ionization probability
             *
             * probability = rate * time step
             * --> for infinitesimal time steps
             *
             * the whole ensemble should then follow
             * P = 1 - exp(-rate * time step) if the laser wavelength is
             * sampled well enough
             */
            float_X probADK     = rateADK * timeStepAU;

            /* ionization condition */
            if (randNr < probADK && chargeState < protonNumber)
            {
                /* set new particle charge state */
                parentIon[boundElectrons_] -= float_X(1.0);
            }

        }
    };

} // namespace ionization
} // namespace particles
} // namespace picongpu
