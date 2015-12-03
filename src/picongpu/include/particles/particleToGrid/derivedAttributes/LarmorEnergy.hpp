/**
 * Copyright 2013-2015 Axel Huebl, Rene Widera, Richard Pausch
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

#include "particles/particleToGrid/derivedAttributes/LarmorEnergy.def"

#include "simulation_defines.hpp"


namespace picongpu
{
namespace particleToGrid
{
namespace derivedAttributes
{

    HDINLINE float1_64
    LarmorEnergy::getUnit() const
    {
        return UNIT_ENERGY;
    }

    template< class T_Particle >
    DINLINE float_X
    LarmorEnergy::operator()( T_Particle& particle ) const
    {
        /* read existing attributes */
        const float3_X mom = particle[momentum_];
        const float3_X mom_mt1 = particle[momentumPrev1_];
        const float_X weighting = particle[weighting_];
        const float_X charge = attribute::getCharge( weighting, particle );
        const float_X mass = attribute::getMass( weighting, particle );

        /* calculate new attribute */
        Gamma<float_X> calcGamma;
        const typename Gamma<float_X>::valueType gamma = calcGamma( mom, mass );
        const float_X gamma2 = gamma * gamma;
        const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

        const float3_X mom_dt = mom - mom_mt1;
        const float_X el_factor = charge * charge
            / (float_X(6.0) * PI * EPS0 *
               c2 * c2 * SPEED_OF_LIGHT * mass * mass);
        const float_X energyLarmor = el_factor * gamma2 * gamma2
            * (math::abs2(mom_dt) -
               math::abs2(math::cross(mom, mom_dt)));

        /* return attribute */
        return energyLarmor;
    }
} /* namespace derivedAttributes */
} /* namespace particleToGrid */
} /* namespace picongpu */
