/* Copyright 2013-2021 Axel Huebl, Rene Widera
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

#include "picongpu/particles/particleToGrid/derivedAttributes/Counter.def"

#include "picongpu/simulation_defines.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                HDINLINE float1_64 Counter::getUnit() const
                {
                    return particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;
                }

                template<class T_Particle>
                DINLINE float_X Counter::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    const float_X weighting = particle[weighting_];

                    /* calculate new attribute */
                    const float_X particleCounter = weighting / particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;

                    /* return attribute */
                    return particleCounter;
                }
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
