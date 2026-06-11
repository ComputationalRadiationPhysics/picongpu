/*
 * SPDX-FileCopyrightText: Rene Widera, Finn-Ole Carstens
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/param/transitionRadiation.param"
#include "picongpu/particles/Manipulate.hpp"
#include "picongpu/plugins/transitionRadiation/GammaMask.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

#include <memory>

namespace picongpu::plugins::transitionRadiation
{
    /** execute the particle filter on a species
     *
     * It is **allowed** to call this function even if the species does not contain
     * the attribute `transitionRadiationMask`.
     * The filter is **not** executed if the species does not contain the attribute `transitionRadiationMask`.
     *
     * @tparam T_Species species type
     * @param species species to be filtered
     */
    template<typename T_Species>
    inline void executeParticleFilter(std::shared_ptr<T_Species>& species, uint32_t const currentStep)
    {
        constexpr bool hasTransitionRadiationFilter
            = pmacc::traits::HasIdentifier<typename T_Species::FrameType, transitionRadiationMask>::type::value;

        if constexpr(hasTransitionRadiationFilter)
            particles::manipulate<picongpu::plugins::transitionRadiation::GammaFilter, T_Species>(currentStep);
    }
} // namespace picongpu::plugins::transitionRadiation
