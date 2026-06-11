/*
 * SPDX-FileCopyrightText: Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/combinedAttributes/AverageAttribute.def"
#include "picongpu/particles/particleToGrid/combinedAttributes/RelativisticDensity.def"
#include "picongpu/traits/frame/GetMass.hpp"

#include <limits>
#include <string>
#include <vector>

namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace combinedAttributes
            {
                template<typename T_Species>
                struct RelativisticDensityOperationImpl
                {
                    /** Functor implementation
                     *
                     * Result overwrites the density value.
                     *
                     * @tparam T_Worker lockstep worker type
                     * @param acc alpaka accelerator
                     * @param density number density value and the result destination
                     * @param energyDensity  energy density value
                     */
                    template<typename T_Worker>
                    HDINLINE void operator()(T_Worker const& worker, float1_X& density, float1_X const& energyDensity)
                        const
                    {
                        float_X const densityPICUnits
                            = density[0] * static_cast<float_X>(sim.unit.typicalNumParticlesPerMacroParticle());
                        // avoid dividing by zero.
                        if(densityPICUnits > std::numeric_limits<float_X>::min())
                        {
                            float_X const averageEnergy = energyDensity[0] / densityPICUnits;
                            float_X const particleMass
                                = picongpu::traits::frame::getMass<typename T_Species::FrameType>();
                            float_X const averageGamma
                                = averageEnergy
                                      / (particleMass * sim.pic.getSpeedOfLight() * sim.pic.getSpeedOfLight())
                                  + 1.0_X;
                            float_X const invAverageGammaSquared = 1.0_X / averageGamma / averageGamma;
                            density *= invAverageGammaSquared;
                        }
                    }
                };

                struct RelativisticDensityDescription
                {
                    HDINLINE float1_64 getUnit() const
                    {
                        // gamma is unitless so the unit stays unchanged
                        return derivedAttributes::Density().getUnit();
                    }

                    HINLINE std::vector<float_64> getUnitDimension() const
                    {
                        return derivedAttributes::Density().getUnitDimension();
                    }

                    HINLINE static std::string getName()
                    {
                        return "relativisticDensity";
                    }
                };

            } // namespace combinedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
