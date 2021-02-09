/* Copyright 2013-2021 Axel Huebl, Rene Widera, Heiko Burau, Marco Garten
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

#include "picongpu/particles/particleToGrid/derivedAttributes/EnergyDensityCutoff.def"

#include "picongpu/simulation_defines.hpp"
#include "picongpu/algorithms/KinEnergy.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                template<class T_ParamClass>
                template<class T_Particle>
                DINLINE float_X EnergyDensityCutoff<T_ParamClass>::operator()(T_Particle& particle) const
                {
                    using ParamClass = T_ParamClass;

                    /* read existing attributes */
                    float_X const weighting = particle[weighting_];
                    float3_X const mom = particle[momentum_];
                    float_X const mass = attribute::getMass(weighting, particle);

                    constexpr float_X INV_CELL_VOLUME = float_X(1.0) / CELL_VOLUME;

                    /* value for energy cut-off */
                    float_X const cutoffMaxEnergy = ParamClass::cutoffMaxEnergy;
                    float_X const cutoff = cutoffMaxEnergy / UNIT_ENERGY * weighting;

                    float_X const kinEnergy = KinEnergy<>()(mom, mass);

                    float_X result(0.);
                    if(kinEnergy < cutoff)
                        result = kinEnergy * INV_CELL_VOLUME;

                    return result;
                }
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
