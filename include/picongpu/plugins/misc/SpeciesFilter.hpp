/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"

#include <boost/mpl/and.hpp>
#include <boost/mpl/bool.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** combines a particle species with a filter
             *
             * @tparam T_Species picongpu::Particle, type of the species
             * @tparam T_Filter pmacc::filter::Interface, type of the filter
             */
            template<typename T_Species, typename T_Filter = particles::filter::All>
            struct SpeciesFilter
            {
                using Filter = T_Filter;
                using Species = T_Species;

                /** name of the filtered species
                 *
                 * @return <speciesName>_<filterName>`
                 */
                static std::string getName()
                {
                    return Species::FrameType::getName() + "_" + Filter::getName();
                }
            };

            /** species without a filter
             *
             * This class fulfills the interface of SpeciesFilter for a species
             * but keeps the species name without adding the filter suffix.
             */
            template<typename T_Species>
            struct UnfilteredSpecies
            {
                using Filter = particles::filter::All;
                using Species = T_Species;

                /** get name of the filtered species
                 *
                 * @return <speciesName>
                 */
                static std::string getName()
                {
                    return Species::FrameType::getName();
                }
            };

            namespace speciesFilter
            {
                /** evaluate if the filter and species combination is valid
                 *
                 * @tparam T_SpeciesFilter SpeciesFilter, type of the filter and species
                 * @return pmacc::mp_bool<>, if the species is eligible for the filter
                 */
                template<typename T_SpeciesFilter>
                using IsEligible = std::bool_constant<
                    particles::traits::SpeciesEligibleForSolver<
                        typename T_SpeciesFilter::Species,
                        typename T_SpeciesFilter::Filter>::type::value
                    && T_SpeciesFilter::Filter::isDeterministic>;
            } // namespace speciesFilter

        } // namespace misc
    } // namespace plugins
} // namespace picongpu
