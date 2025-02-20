/* Copyright 2023-2025 Tapish Narwal
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

#include "picongpu/plugins/binning/utility.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace picongpu
{
    namespace plugins::binning
    {
        struct AllParticles
        {
            HDINLINE bool operator()(auto const& worker, auto const& domainInfo, auto const& particle) const
            {
                return true;
            }
        };

        template<typename TSpecies, typename TFilter = AllParticles>
        struct FilteredSpecies
        {
            using species_type = TSpecies;
            using filter_type = TFilter;

            TSpecies species;
            TFilter filter;

            FilteredSpecies(TSpecies species, TFilter filter) noexcept : species(species), filter(filter)
            {
            }

            FilteredSpecies(TSpecies species) noexcept : species(species), filter()
            {
            }
        };

        template<typename T>
        concept IsFilteredSpecies = requires
        {
            typename T::species_type;
            typename T::filter_type;
        };

        /**
         * Function to create a tuple of FilteredSpecies
         * If you pass in a type which is not a FilteredSpecies, it is assumed to be a regular Species type, and
         * a trivial AllParticle filter is used with it, which allows all particles through without filtering
         */
        template<typename... Args>
        constexpr auto createSpeciesTuple(Args&&... args)
        {
            return createTuple(
                (IsFilteredSpecies<Args> ? std::forward<Args>(args) : FilteredSpecies{std::forward<Args>(args)})...);
        }

    } // namespace plugins::binning
} // namespace picongpu
