/**
 * Copyright 2016 Axel Huebl
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

#include "particles/particleToGrid/derivedAttributes/EnergyPerpendicular.def"

#include "simulation_defines.hpp"


namespace picongpu
{
namespace particleToGrid
{
namespace derivedAttributes
{

    template< size_t T_direction>
    HDINLINE float1_64
    EnergyPerpendicular<T_direction>::getUnit() const
    {
        return 1.0;
    }

    template< size_t T_direction>
    DINLINE float_X
    EnergyPerpendicular<T_direction>::getEnergy( const float3_X mom, const float_X mass ) const
    {
        const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        Gamma<float_X> calcGamma;
        const typename Gamma<float_X>::valueType gamma =
            calcGamma( mom, mass );

        const float_X energy = ( gamma <= float_X(GAMMA_THRESH) ) ?
            math::abs2(mom) / ( float_X(2.0) * mass ) :   /* non-relativistic */
            (gamma - float_X(1.0)) * mass * c2;           /* relativistic     */

        return energy;
    }

    template< size_t T_direction>
    template< class T_Particle >
    DINLINE float_X
    EnergyPerpendicular<T_direction>::operator()( T_Particle& particle ) const
    {
        /* read existing attributes */
        const float_X weighting = particle[weighting_];
        const float3_X mom = particle[momentum_];
        const float_X mass = attribute::getMass( weighting, particle );
        float3_X momLong = float3_X::create(0.0);
        momLong[T_direction] = particle[momentum_][T_direction];
        float3_X momPerp = particle[momentum_];
        momPerp[T_direction] = float_X(0.0);

        /* calculate new attribute */
        const float_X energyPerp = getEnergy( momPerp, mass );
        const float_X energyLong = getEnergy( momLong, mass );

        const float_X energyPerpOverLong = energyPerp / energyLong;

        /* return attribute */
        return energyPerpOverLong;
    }
} // namespace derivedAttributes
} // namespace particleToGrid
} // namespace picongpu
