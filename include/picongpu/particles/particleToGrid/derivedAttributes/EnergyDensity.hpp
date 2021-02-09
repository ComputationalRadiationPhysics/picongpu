/* Copyright 2013-2021 Axel Huebl, Rene Widera, Heiko Burau
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

#include "picongpu/particles/particleToGrid/derivedAttributes/EnergyDensity.def"

#include "picongpu/simulation_defines.hpp"
#include "picongpu/algorithms/KinEnergy.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                HDINLINE float1_64 EnergyDensity::getUnit() const
                {
                    constexpr float_64 UNIT_VOLUME = (UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH);
                    return UNIT_ENERGY / UNIT_VOLUME;
                }

                template<class T_Particle>
                DINLINE float_X EnergyDensity::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    const float_X weighting = particle[weighting_];
                    const float3_X mom = particle[momentum_];
                    const float_X mass = attribute::getMass(weighting, particle);

                    constexpr float_X INV_CELL_VOLUME = float_X(1.0) / CELL_VOLUME;

                    return KinEnergy<>()(mom, mass) * INV_CELL_VOLUME;
                }
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
