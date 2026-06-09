/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2021-2024 Pawel Ordyna
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
