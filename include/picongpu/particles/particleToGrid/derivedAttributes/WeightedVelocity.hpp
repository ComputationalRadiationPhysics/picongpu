/*
 * SPDX-FileCopyrightText: Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/algorithms/Velocity.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/IsWeighted.hpp"
#include "picongpu/particles/particleToGrid/derivedAttributes/Momentum.def"
#include "picongpu/particles/particleToGrid/derivedAttributes/WeightedVelocity.def"
#include "picongpu/traits/attribute/GetMass.hpp"

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
                template<typename T_Particle>
                DINLINE float_X WeightedVelocity<T_direction>::operator()(T_Particle& particle) const
                {
                    float_X const weighting = particle[weighting_];
                    float_X const mass = picongpu::traits::attribute::getMass(weighting, particle);

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
