/* Copyright 2013-2023 Axel Huebl, Rene Widera, Heiko Burau
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/EnergyDensity.def"
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"

#include <type_traits>


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
                    constexpr float_64 UNIT_VOLUME = (sim.unit.length() * sim.unit.length() * sim.unit.length());
                    return sim.unit.energy() / UNIT_VOLUME;
                }

                template<class T_Particle>
                DINLINE float_X EnergyDensity::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    const float_X weighting = particle[weighting_];
                    const float3_X mom = particle[momentum_];
                    const float_X mass = attribute::getMass(weighting, particle);

                    constexpr float_X invCellVolume = float_X(1.0) / sim.pic.getCellSize().productOfComponents();

                    return KinEnergy<>()(mom, mass) * invCellVolume;
                }

                //! Energy density is weighted
                template<>
                struct IsWeighted<EnergyDensity> : std::true_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
