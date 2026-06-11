/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
