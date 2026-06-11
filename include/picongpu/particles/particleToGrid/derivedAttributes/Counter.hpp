/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/Counter.def"
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
                template<class T_Particle>
                DINLINE float_X Counter::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    float_X const weighting = particle[weighting_];

                    /* calculate new attribute */
                    float_X const particleCounter
                        = weighting / static_cast<float_X>(sim.unit.typicalNumParticlesPerMacroParticle());

                    /* return attribute */
                    return particleCounter;
                }

                //! Counter is weighted (as it is a count of real particles)
                template<>
                struct IsWeighted<Counter> : std::true_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
