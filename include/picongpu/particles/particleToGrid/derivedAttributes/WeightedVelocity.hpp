/* Copyright 2022-2023 Pawel Ordyna
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

#include "picongpu/algorithms/Velocity.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/Momentum.def"

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
                HDINLINE float1_64 WeightedVelocity<T_direction>::getUnit() const
                {
                    return sim.unit.speed();
                }

                template<size_t T_direction>
                template<typename T_Particle>
                DINLINE float_X WeightedVelocity<T_direction>::operator()(T_Particle& particle) const
                {
                    const float_X weighting = particle[weighting_];
                    const float_X mass = picongpu::traits::attribute::getMass(weighting, particle);

                    return weighting * (picongpu::Velocity{}(particle[momentum_], mass))[T_direction];
                }

                //! Component of momentum is weighted
                template<size_t T_direction>
                struct IsWeighted<WeightedVelocity<T_direction>> : std::true_type
                {
                };

            } // namespace derivedAttributes
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
