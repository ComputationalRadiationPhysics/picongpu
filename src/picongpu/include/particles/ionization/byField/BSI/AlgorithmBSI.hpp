/**
 * Copyright 2014 Marco Garten
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
#include "traits/attribute/GetChargeState.hpp"

/** IONIZATION ALGORITHM 
 * - implements the calculation of ionization probability and changes charge states
 *   by decreasing the number of bound electrons
 * - is called with the IONIZATION MODEL, specifically by setting the flag in @see speciesDefinition.param */

namespace picongpu
{
namespace particles
{
namespace ionization
{

    /** \struct AlgorithmBSI
     * \brief calculation for the Barrier Suppression Ionization model */
    struct AlgorithmBSI
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
        operator()( const BType bField, const EType eField, ParticleType& parentIon )
        {

            const float_X protonNumber = GetAtomicNumbers<ParticleType>::type::numberOfProtons;
            float_X chargeState = attribute::getChargeState(parentIon);
            
            /* ionization condition */
            if (math::abs(eField)*UNIT_EFIELD >= SI::IONIZATION_EFIELD && chargeState < protonNumber)
            {
                /* set new particle charge state */
                parentIon[boundElectrons_] -= float_X(1.0);
            }

        }
    };

} // namespace ionization
} // namespace particles
} // namespace picongpu
