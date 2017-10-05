/* Copyright 2013-2017 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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

#include "picongpu/simulation_defines.hpp"


namespace picongpu
{
namespace particles
{
namespace startPosition
{
namespace detail
{

    /** Calculate the weighting per macro-particle in a cell
     *
     * Note: In the density regions where the weighting of macro particles would
     * violate the user-specified MIN_WEIGHTING, we reduce the number of
     * macro particles per cell to still initialize particles
     * (see particles.param).
     *
     * This calculates the number of macro particles and the weighting per macro
     * particle with respect to MIN_WEIGHTING.
     */
    struct WeightMacroParticles
    {
        /** get number of and the weighting per macro particle(s)
         *
         * @param realParticlesPerCell number of real particles per cell
         * @param macroParticlesPerCell maximum number of macro particles per cell
         * @param[out] weighting weighting per macro particle
         * @return number of macro particles per cell with respect to
         *         MIN_WEIGHTING, range: [0;macroParticlesPerCell]
         */
        DINLINE uint32_t
        operator()(
            float_X const realParticlesPerCell,
            uint32_t numMacroParticles,
            float_X & weighting
        ) const
        {
            weighting = float_X( 0.0 );
            if( numMacroParticles > 0u )
                weighting = realParticlesPerCell / float_X( numMacroParticles );

            while(
                weighting < MIN_WEIGHTING &&
                numMacroParticles > 0u
            )
            {
                --numMacroParticles;
                if( numMacroParticles > 0u )
                    weighting = realParticlesPerCell / float_X( numMacroParticles );
                else
                    weighting = float_X( 0.0 );
            }

            return numMacroParticles;
        }
    };

} // namespace detail
} // namespace startPosition
} // namespace particles
} // namespace picongpu
