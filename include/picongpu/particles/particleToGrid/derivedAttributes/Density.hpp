/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/Density.def"
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
                DINLINE float_X Density::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    float_X const weighting = particle[weighting_];

                    /* calculate new attribute */
                    float_X const particleDensity
                        = weighting
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
