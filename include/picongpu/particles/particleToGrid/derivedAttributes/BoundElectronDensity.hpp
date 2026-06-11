/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/BoundElectronDensity.def"
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
                template<class T_Particle>
                DINLINE float_X BoundElectronDensity::operator()(T_Particle& particle) const
                {
                    // read existing attributes
                    float_X const weighting = particle[weighting_];
                    float_X const boundElectrons = particle[boundElectrons_];

                    // calculate new attribute
                    float_X const boundElectronDensity
                        = weighting * boundElectrons
                          / (static_cast<float_X>(sim.unit.typicalNumParticlesPerMacroParticle())
                             * sim.pic.getCellSize().productOfComponents());

                    return boundElectronDensity;
                }

                //! Bound electron density is weighted
                template<>
                struct IsWeighted<BoundElectronDensity> : std::true_type
                {
                };
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
