/* Copyright 2016-2021 Axel Huebl
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

#include "picongpu/particles/particleToGrid/derivedAttributes/MidCurrentDensityComponent.def"

#include "picongpu/simulation_defines.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                template<size_t T_direction>
                HDINLINE float1_64 MidCurrentDensityComponent<T_direction>::getUnit() const
                {
                    const float_64 UNIT_AREA = UNIT_LENGTH * UNIT_LENGTH;
                    return UNIT_CHARGE / (UNIT_TIME * UNIT_AREA);
                }

                template<size_t T_direction>
                template<class T_Particle>
                DINLINE float_X MidCurrentDensityComponent<T_direction>::operator()(T_Particle& particle) const
                {
                    /* read existing attributes */
                    const float_X weighting = particle[weighting_];
                    const float_X charge = attribute::getCharge(weighting, particle);
                    const float3_X mom = particle[momentum_];
                    const float_X momCom = mom[T_direction];
                    const float_X mass = attribute::getMass(weighting, particle);

                    /* calculate new attribute */
                    Gamma<float_X> calcGamma;
                    const typename Gamma<float_X>::valueType gamma = calcGamma(mom, mass);

                    /* calculate new attribute */
                    const float_X particleCurrentDensity = charge / CELL_VOLUME * /* rho */
                        momCom / (gamma * mass); /* v_component */

                    /* return attribute */
                    return particleCurrentDensity;
                }
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
