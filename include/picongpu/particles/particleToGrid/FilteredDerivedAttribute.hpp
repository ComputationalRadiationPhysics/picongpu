/*
 * SPDX-FileCopyrightText: Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
