/* Copyright 2022 Pawel Ordyna
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

#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/particles/particleToGrid/ComputeFieldValue.hpp"
#include "picongpu/particles/particleToGrid/combinedAttributes/CombinedAttributes.def"
#include "picongpu/plugins/misc/SpeciesFilter.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            namespace detail
            {
                //! Wrap SpeciesFilter around a species type
                template<typename T_Species>
                struct GetSpeciesFilter
                {
                    // For unfiltered species just use SpeciesFilter with the default filter = all
                    using type = typename plugins::misc::SpeciesFilter<T_Species>;
                };

                // Ignore the wrap and just return the same type when the species are already filtered
                template<typename... T>
                struct GetSpeciesFilter<plugins::misc::SpeciesFilter<T...>>
                {
                    using type = typename plugins::misc::SpeciesFilter<T...>;
                };


                /** Modifies a FieldTmp by calculating a derived attribute and applying a binary operation to both
                 *
                 * Operates on two tmp fields, one should be initialized before, the second one is computed from a
                 * (filtered) species using a derived attribute.
                 *
                 * @tparam T_Op the binary operation to use.
                 * @tparam T_FilteredSpecies particle species used to derived the second quantity, combined with a
                 * species. Should be of type plugins::misc::SpeciesFilter .
                 * @tparam T_DerivedAttribute a derived attribute to be used in computation of the second field
                 */
                template<typename T_Op, typename T_FilteredSpecies, typename T_DerivedAttribute>
                struct OpWithNextField
                {
                    /** Functor implementation
                     *
                     * @param fieldTmp1 the first field (should be previously initialized)
                     * @param fieldTmp2 the second field (will be overridden  by the derived quantity)
                     * @param currentStep current simulation time step
                     * @param extraSlotNr a field tmp slot that can be used for fieldTmp2 calculation in case
                     *  it needs two slots (true if T_DerivedAttribute is a combined attribute).
                     */
                    HINLINE void operator()(
                        FieldTmp& fieldTmp1,
                        FieldTmp& fieldTmp2,
                        uint32_t const& currentStep,
                        uint32_t const& extraSlotNr) const
                    {
                        using DeriveOperation = particles::particleToGrid::
                            GetFieldTmpOperationForFilteredSpecies_t<T_FilteredSpecies, T_DerivedAttribute>;

                        using Solver = typename DeriveOperation::Solver;
                        using Species = typename DeriveOperation::Species;
                        using Filter = typename DeriveOperation::Filter;
                        auto event
                            = particles::particleToGrid::ComputeFieldValue<CORE + BORDER, Solver, Species, Filter>()(
                                fieldTmp2,
                                currentStep,
                                extraSlotNr);
                        // wait for unfinished asynchronous communication
                        if(event.has_value())
                            eventSystem::setTransactionEvent(*event);
                        fieldTmp1.template modifyByField<CORE + BORDER, T_Op>(fieldTmp2);
                    }
                };
            } // namespace detail

            /** Fold transformation of derived fields from a list of (filtered) species
             *
             * @tparam AREA area to perform the transformation
             * @tparam T_SpeciesSeq a sequence of (filtered) species to go over. Each element has to be either
             *  a species type or plugins::misc::SpeciesFilter.
             * @tparam T_Op a binary operation that combines the intermediary result of the fold operation
             *  with the attribute derived from the currently processed species
             *  So e.g. with T_Op = Add the result of the fold is a sum over the derived attribute from all the species
             *  in the sequence.
             * @tparam T_DerivedAttribute the derived attribute that should be calculated for all species in the list
             */
            template<uint32_t AREA, typename T_SpeciesSeq, typename T_Op, typename T_DerivedAttribute>
            struct FoldDeriveFields
            {
                /** Functor implementation
                 *
                 * @param fieldTmp1 the tmp field that is used to calculate the result
                 * @param currentStep current simulation step
                 * @param extraSlot an extra FieldTmp slot that can be used for computing the first derived field
                 *  in the sequence if T_DerivedAttribute is a combined attribute. And/Or it is used to store the
                 *  (intermediary) second field that is then combined with fieldTmp1 (only needed when the list of
                 *  species has more than one element) . If T_DerivedAttribute is a combined attribute the
                 *  extraSlot + 1 is used in the calculation of the second field. In summary the required number
                 *  of additional tmp slots (except for fieldTmp1) is:
                 *      ####################################################
                 *      |                    | #species > 1 | #species = 1 |
                 *      | combined attribute |      2       |     1        |
                 *      | simple attribute   |      1       |     0        |
                 *      ####################################################
                 */
                void operator()(FieldTmp& fieldTmp1, uint32_t const& currentStep, uint32_t const& extraSlot) const
                {
                    using FilteredSpeciesSeq =
                        typename bmpl::transform<T_SpeciesSeq, detail::GetSpeciesFilter<bmpl::_1>>::type;

                    using FirstFilteredSpecies = typename bmpl::at_c<FilteredSpeciesSeq, 0>::type;
                    using RemainingFilteredSpecies = typename bmpl::pop_front<FilteredSpeciesSeq>::type;


                    using DeriveOperation = particles::particleToGrid::
                        GetFieldTmpOperationForFilteredSpecies_t<FirstFilteredSpecies, T_DerivedAttribute>;

                    using Solver = typename DeriveOperation::Solver;
                    using Filter = typename DeriveOperation::Filter;
                    using Species = typename DeriveOperation::Species;
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto event = particles::particleToGrid::ComputeFieldValue<AREA, Solver, Species, Filter>()(
                        fieldTmp1,
                        currentStep,
                        extraSlot);
                    // wait for unfinished asynchronous communication
                    if(event.has_value())
                        eventSystem::setTransactionEvent(*event);

                    if constexpr(!bmpl::empty<RemainingFilteredSpecies>::value)
                    {
                        auto fieldTmp2 = dc.get<FieldTmp>(FieldTmp::getUniqueId(extraSlot));
                        pmacc::meta::ForEach<
                            RemainingFilteredSpecies,
                            detail::OpWithNextField<T_Op, bmpl::_1, T_DerivedAttribute>>{}(
                            fieldTmp1,
                            *fieldTmp2,
                            currentStep,
                            extraSlot + 1u);
                    }
                }
            };
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
