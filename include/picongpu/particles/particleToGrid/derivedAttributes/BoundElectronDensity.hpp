/* Copyright 2015-2019 Axel Huebl
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

#include "picongpu/particles/particleToGrid/derivedAttributes/ChargeDensity.def"
#include "picongpu/simulation_defines.hpp"


namespace picongpu
{
namespace particles
{
namespace particleToGrid
{
namespace derivedAttributes
{

    HDINLINE float1_64
    BoundElectronDensity::getUnit() const
    {
        constexpr float_64 UNIT_VOLUME = UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH;
        return particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE / UNIT_VOLUME;
    }

    template< class T_Particle >
    DINLINE float_X
    BoundElectronDensity::operator()( T_Particle& particle ) const
    {
        // read existing attributes
        float_X const weighting = particle[ weighting_ ];
        float_X const boundElectrons = particle[ boundElectrons_ ];

        // calculate new attribute
        float_X const boundElectronDensity = weighting * boundElectrons /
            ( particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE * CELL_VOLUME );

        return boundElectronDensity;
    }
} // namespace derivedAttributes
} // namespace particleToGrid
} // namespace particles
} // namespace picongpu
