/* Copyright 2017-2023 Axel Huebl
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
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/MacroCounter.def"

#include <type_traits>


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                template<class T_Particle>
                DINLINE float_X MacroCounter::operator()(T_Particle& particle) const
                {
                    /* return attribute */
                    return 1.0;
                }

                //! Macroparticle counter is not weighted
                template<>
                struct IsWeighted<MacroCounter> : std::false_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
