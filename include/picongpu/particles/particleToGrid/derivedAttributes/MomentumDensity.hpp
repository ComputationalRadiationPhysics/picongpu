/* Copyright 2016-2023 Axel Huebl, Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/MomentumDensity.def"

#include <type_traits>


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace derivedAttributes
            {
                template<size_t T_direction>
                HDINLINE float1_64 MomentumDensity<T_direction>::getUnit() const
                {
                    constexpr float_64 UNIT_VOLUME = (UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH);
                    return UNIT_MASS * UNIT_SPEED / UNIT_VOLUME;
                }

                template<size_t T_direction>
                template<typename T_Particle>
                DINLINE float_X MomentumDensity<T_direction>::operator()(T_Particle& particle) const
                {
                    constexpr float_X INV_CELL_VOLUME = float_X(1.0) / CELL_VOLUME;
                    return particle[momentum_][T_direction] * INV_CELL_VOLUME;
                }

                //! Density of component of momentum is weighted
                template<size_t T_direction>
                struct IsWeighted<MomentumDensity<T_direction>> : std::true_type
                {
                };

            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
