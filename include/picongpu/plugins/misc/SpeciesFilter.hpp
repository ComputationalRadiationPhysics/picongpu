/* Copyright 2017-2021 Rene Widera
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
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/particles/filter/filter.def"


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
                 * @return ::type boost::mpl::bool_<>, if the species is eligible for the filter
                 */
                template<typename T_SpeciesFilter>
                struct IsEligible
                {
                    using type = typename particles::traits::SpeciesEligibleForSolver<
                        typename T_SpeciesFilter::Species,
                        typename T_SpeciesFilter::Filter>::type;
                };
            } // namespace speciesFilter

        } // namespace misc
    } // namespace plugins
} // namespace picongpu
