/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/startPosition/detail/WeightMacroParticles.def"

namespace picongpu::particles::startPosition::detail
{
    HDINLINE uint32_t WeightMacroParticles::operator()(
        float_X const realParticlesPerCell,
        uint32_t numMacroParticles,
        float_X& weighting) const
    {
        PMACC_CASSERT_MSG(__MIN_WEIGHTING_must_be_greater_than_zero, MIN_WEIGHTING > float_X(0.0));
        weighting = float_X(0.0);
        float_X const maxParPerCell = realParticlesPerCell / MIN_WEIGHTING;
        numMacroParticles = pmacc::math::float2int_rd(math::min(float_X(numMacroParticles), maxParPerCell));
        if(numMacroParticles > 0u)
            weighting = realParticlesPerCell / float_X(numMacroParticles);

        return numMacroParticles;
    }
} // namespace picongpu::particles::startPosition::detail
