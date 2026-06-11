/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu::plugins::radiation
{
    /** get the value of the particle attribute `radiationMask`
     *
     * Allow to read out the value of the attribute `radiationMask` also if
     * it is not defined for the particle.
     *
     * @tparam T_Particle particle type
     * @param particle valid particle
     * @return particle attribute value `radiationMask`, always `true` if attribute `radiationMask` is not
     * defined
     */
    template<typename T_Particle>
    HDINLINE bool getRadiationMask(T_Particle const& particle)
    {
        constexpr bool hasRadiationMask
            = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, radiationMask>::type::value;
        if constexpr(hasRadiationMask)
            return particle[picongpu::radiationMask_];
        else
            return true;
    }
} // namespace picongpu::plugins::radiation
