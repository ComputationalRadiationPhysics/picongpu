/* Copyright 2015-2023 Axel Huebl
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/ChargeDensity.def"
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
                HDINLINE float1_64 Density::getUnit() const
                {
                    const float_64 UNIT_VOLUME = (sim.unit.length() * sim.unit.length() * sim.unit.length());
                    return sim.unit.typicalNumParticlesPerMacroParticle() / UNIT_VOLUME;
                }

                template<class T_Particle>
                DINLINE float_X Density::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    const float_X weighting = particle[weighting_];

                    /* calculate new attribute */
                    const float_X particleDensity = weighting
                        / (static_cast<float_X>(sim.unit.typicalNumParticlesPerMacroParticle())
                           * sim.pic.getCellSize().productOfComponents());

                    /* return attribute */
                    return particleDensity;
                }

                //! Density is weighted
                template<>
                struct IsWeighted<Density> : std::true_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
