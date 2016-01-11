/**
 * Copyright 2013-2016 Axel Huebl, Rene Widera
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

#include "particles/particleToGrid/derivedAttributes/Energy.def"

#include "simulation_defines.hpp"


namespace picongpu
{
namespace particleToGrid
{
namespace derivedAttributes
{

    HDINLINE float1_64
    Energy::getUnit() const
    {
        return UNIT_ENERGY;
    }

    template< class T_Particle >
    DINLINE float_X
    Energy::operator()( T_Particle& particle ) const
    {
        /* read existing attributes */
        const float_X weighting = particle[weighting_];
        const float3_X mom = particle[momentum_];
        const float_X mass = attribute::getMass( weighting, particle );

        /* calculate new attribute */
        Gamma<float_X> calcGamma;
        const typename Gamma<float_X>::valueType gamma = calcGamma( mom, mass );
        const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

        const float_X energy = ( gamma <= float_X(GAMMA_THRESH) ) ?
            math::abs2(mom) / ( float_X(2.0) * mass ) :   /* non-relativistic */
            (gamma - float_X(1.0)) * mass * c2;           /* relativistic     */

        /* return attribute */
        return energy;
    }
} /* namespace derivedAttributes */
} /* namespace particleToGrid */
} /* namespace picongpu */
