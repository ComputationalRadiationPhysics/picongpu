/* Copyright 2023 Brian Marre
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

/** @file chooseTransition sub-stage of atomicPhysics
 *
 * randomly decide on one known transition for each macro-ion from it's current atomic state
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/kernel/ChooseTransition.kernel"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalTimeRemainingField.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"

/// @todo find reference to pmacc RNGfactories files, Brian Marre, 2023

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2::stage
{
    /** atomic physics sub-stage for choosing one known transition for each macro-ion
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct ChooseTransition
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        using DistributionInt = pmacc::random::distributions::Uniform<uint32_t>;
        using RngFactoryInt = particles::functor::misc::Rng<DistributionInt>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;
            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            using SpeciesConfigNumberType = typename AtomicDataType::ConfigNumber;

            auto& localTimeRemainingField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<
                    picongpu::MappingDesc>>("LocalTimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            RngFactoryInt rngFactory = RngFactoryInt{currentStep};

            PMACC_LOCKSTEP_KERNEL(
                picongpu::particles::atomicPhysics2::kernel::ChooseTransitionKernel<SpeciesConfigNumberType>(),
                workerCfg)
            (mapper.getGridDim())(
                mapper,
                rngFactory,
                localTimeRemainingField.getDeviceDataBox(),
                ions.getDeviceParticlesBox(),
                atomicData.template getChargeStateOrgaDataBox<false>(),
                atomicData.template getAtomicStateDataDataBox<false>(),
                atomicData.template getTransitionSelectionDataBox<false>());
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage
