/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
        concept IsFilteredSpecies = requires {
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
