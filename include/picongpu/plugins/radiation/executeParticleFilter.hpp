/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/param/radiation.param"
#include "picongpu/particles/Manipulate.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu::plugins::radiation
{
    /** execute the particle filter on a species
     *
     * It is **allowed** to call this function even if the species does not contain
     * the attribute `radiationMask`.
     * The filter is **not** executed if the species does not contain the attribute `radiationMask`.
     *
     * @tparam T_Species species type
     * @param species species to be filtered
     */
    template<typename T_Species>
    inline void executeParticleFilter(
        [[maybe_unused]] std::shared_ptr<T_Species>& species,
        [[maybe_unused]] uint32_t const currentStep)
    {
        constexpr bool hasRadiationFilter
            = pmacc::traits::HasIdentifier<typename T_Species::FrameType, radiationMask>::type::value;

        if constexpr(hasRadiationFilter)
        {
            auto executeFilter
                = particles::manipulate<picongpu::plugins::radiation::RadiationParticleFilter, T_Species>(currentStep);
        }
    }
} // namespace picongpu::plugins::radiation
