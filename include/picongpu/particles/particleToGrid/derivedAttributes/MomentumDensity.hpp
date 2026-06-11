/*
 * SPDX-FileCopyrightText: Axel Huebl, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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
                template<typename T_Particle>
                DINLINE float_X MomentumDensity<T_direction>::operator()(T_Particle& particle) const
                {
                    constexpr float_X INV_CELL_VOLUME = float_X(1.0) / sim.pic.getCellSize().productOfComponents();
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
