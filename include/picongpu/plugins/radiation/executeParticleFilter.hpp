/* Copyright 2017-2023 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

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
        [[maybe_unused]] const uint32_t currentStep)
    {
        constexpr bool hasRadiationFilter
            = pmacc::traits::HasIdentifier<typename T_Species::FrameType, radiationMask>::type::value;

        if constexpr(hasRadiationFilter)
        {
            auto executeFilter
                = particles::Manipulate<picongpu::plugins::radiation::RadiationParticleFilter, T_Species>{};
            executeFilter(currentStep);
        }
    }
} // namespace picongpu::plugins::radiation
