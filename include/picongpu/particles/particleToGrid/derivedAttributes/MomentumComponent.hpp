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

#include "picongpu/particles/particleToGrid/derivedAttributes/MomentumComponent.def"

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
                HDINLINE float1_64 MomentumComponent<T_direction>::getUnit() const
                {
                    return 1.0;
                }

                template<size_t T_direction>
                template<class T_Particle>
                DINLINE float_X MomentumComponent<T_direction>::operator()(T_Particle& particle) const
                {
                    // read existing attributes
                    const float3_X mom = particle[momentum_];

                    // calculate new attribute: |p| and p.[x|y|z]
                    const float_X momAbs = math::abs(mom);
                    const float_X momCom = mom[T_direction];

                    // total momentum == 0 then perpendicular measure shall be zero, too
                    // values: [-1.:1.]
                    const float_X momComOverTotal = (momAbs > float_X(0.)) ? momCom / momAbs : float_X(0.);

                    // return attribute
                    return momComOverTotal;
                }
            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
