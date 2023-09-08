/* Copyright 2014-2023 Pawel Ordyna
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

#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/particleToGrid/derivedAttributes/DerivedAttributes.def"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            /** Combine derived field and a particle filter
             *
             * This struct is used for combining  SpeciesEligibleForSolver tests for a solver and a particle filter
             * in one. This enables generating only the TmpField operations that pass both tests.
             * @tparam T_DerivedAttribute derived attribute used in a `FieldTmpOperation`
             * @tparam T_Filter particle filter used in a `FieldTmpOperation`
             */
            template<typename T_DerivedAttribute, typename T_Filter>
            struct FilteredDerivedAttribute
            {
                using DerivedAttribute = T_DerivedAttribute;
                using Filter = T_Filter;
            };
        } // namespace particleToGrid
        namespace traits
        {
            template<typename T_Species, typename T_DerivedAttribute, typename T_Filter>
            struct SpeciesEligibleForSolver<
                T_Species,
                particleToGrid::FilteredDerivedAttribute<T_DerivedAttribute, T_Filter>>
            {
                using EligibleForDerivedAttribute =
                    typename particles::traits::SpeciesEligibleForSolver<T_Species, T_DerivedAttribute>::type;
                using EligibleForFilter =
                    typename particles::traits::SpeciesEligibleForSolver<T_Species, T_Filter>::type;
                using type = pmacc::mp_and<EligibleForDerivedAttribute, EligibleForFilter>;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
