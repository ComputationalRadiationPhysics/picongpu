/* Copyright 2017-2021 Axel Huebl
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

#include "picongpu/particles/flylite/helperFields/LocalDensity.hpp"
#include "picongpu/particles/flylite/helperFields/LocalDensity.kernel"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"

// pmacc
#include <pmacc/types.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/meta/ForEach.hpp>

#include <string>
#include <memory>


namespace picongpu
{
    namespace particles
    {
        namespace flylite
        {
            namespace helperFields
            {
                namespace detail
                {
                    /** Average a group of species to a local density
                     *
                     * Takes a single species and fills the LocalDensity with it.
                     *
                     * @tparam T_SpeciesType a picongpu::Particles class with a particle species
                     */
                    template<typename T_SpeciesType>
                    struct AddSingleDensity
                    {
                        using SpeciesType = T_SpeciesType;
                        using FrameType = typename SpeciesType::FrameType;
                        using ShapeType = typename GetShape<SpeciesType>::type;

                        /** Functor
                         *
                         * @param currentStep the current time step
                         * @param fieldTmp a slot of FieldTmp to add a density to
                         */
                        void operator()(uint32_t currentStep, std::shared_ptr<FieldTmp>& fieldTmp)
                        {
                            DataConnector& dc = Environment<>::get().DataConnector();

                            // load particle without copy particle data to host
                            auto speciesTmp = dc.get<SpeciesType>(FrameType::getName(), true);

                            using Density = particleToGrid::
                                ComputeGridValuePerFrame<ShapeType, particleToGrid::derivedAttributes::Density>;
                            fieldTmp->template computeValue<CORE + BORDER, Density>(*speciesTmp, currentStep);

                            dc.releaseData(FrameType::getName());
                        }
                    };
                } // namespace detail
                /** Average a group of species to a local density
                 *
                 * Takes a list of species and fills the LocalDensity with it.
                 * Ideally executed for a list of electron species or an ion species.
                 *
                 * @tparam T_SpeciesList sequence of picongpu::Particles to create a
                 *                       local density from
                 */
                template<typename T_SpeciesList>
                struct FillLocalDensity
                {
                    using SpeciesList = T_SpeciesList;

                    /** Functor
                     *
                     * @param currentStep the current time step
                     * @param speciesGroup naming for the group of species in T_SpeciesList
                     */
                    void operator()(uint32_t currentStep, std::string const& speciesGroup)
                    {
                        // generating a density requires at least one slot in FieldTmp
                        PMACC_CASSERT_MSG(
                            _please_allocate_at_least_one_FieldTmp_in_memory_param,
                            fieldTmpNumSlots > 0);

                        DataConnector& dc = Environment<>::get().DataConnector();

                        // load FieldTmp without copy data to host and zero it
                        auto fieldTmp = dc.get<FieldTmp>(FieldTmp::getUniqueId(0), true);
                        using DensityValueType = typename FieldTmp::ValueType;
                        fieldTmp->getGridBuffer().getDeviceBuffer().setValue(DensityValueType::create(0.0));

                        // add density of each species in list to FieldTmp
                        meta::ForEach<SpeciesList, detail::AddSingleDensity<bmpl::_1>> addSingleDensity;
                        addSingleDensity(currentStep, fieldTmp);

                        /* create valid density in the BORDER region
                         * note: for average != supercell multiples the GUARD of fieldTmp
                         *       also needs to be filled in the communication above
                         */
                        EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
                        __setTransactionEvent(fieldTmpEvent);

                        /* average summed density in FieldTmp down to local resolution and
                         * write in new field
                         */
                        auto nlocal = dc.get<LocalDensity>(helperFields::LocalDensity::getName(speciesGroup), true);
                        constexpr uint32_t numWorkers
                            = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                        PMACC_KERNEL(helperFields::KernelAverageDensity<numWorkers>{})
                        (
                            // one block per averaged density value
                            nlocal->getGridBuffer().getGridLayout().getDataSpaceWithoutGuarding(),
                            numWorkers)(
                            // start in border (jump over GUARD area)
                            fieldTmp->getDeviceDataBox().shift(SuperCellSize::toRT() * GuardSize::toRT()),
                            // start in border (has no GUARD area)
                            nlocal->getGridBuffer().getDeviceBuffer().getDataBox());

                        // release fields
                        dc.releaseData(FieldTmp::getUniqueId(0));
                        dc.releaseData(helperFields::LocalDensity::getName(speciesGroup));
                    }
                };

            } // namespace helperFields
        } // namespace flylite
    } // namespace particles
} // namespace picongpu
