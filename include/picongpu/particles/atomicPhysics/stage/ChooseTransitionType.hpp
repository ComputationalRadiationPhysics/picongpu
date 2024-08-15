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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file chooseTransitionType sub-stage of atomicPhysics
 *
 * randomly choose one transitionType for each macro-ion
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/kernel/ChooseTransitionType.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"

/// @todo find reference to pmacc RNGfactories files, Brian Marre, 2023

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomic physics sub-stage for choosing one active transitionType for each macro-ion
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct ChooseTransitionType
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        using DistributionFloat = pmacc::random::distributions::Uniform<float_X>;
        using RngFactoryFloat = particles::functor::misc::Rng<DistributionFloat>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            auto& localTimeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                "LocalTimeRemainingField");
            auto& localTimeStepField = *dc.get<
                picongpu::particles::atomicPhysics ::localHelperFields::LocalTimeStepField<picongpu::MappingDesc>>(
                "LocalTimeStepField");
            using RateCacheType = typename picongpu::particles::atomicPhysics::localHelperFields::
                LocalRateCacheField<picongpu::MappingDesc, IonSpecies>::entryType;
            auto& localRateCacheField = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                                                    LocalRateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_localRateCacheField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());
            RngFactoryFloat rngFactoryFloat = RngFactoryFloat{currentStep};

            using ChooseTransitionTypeKernel =
                typename picongpu::particles::atomicPhysics::kernel::ChooseTransitionTypeKernel<
                    RateCacheType,
                    AtomicDataType::switchElectronicExcitation,
                    AtomicDataType::switchElectronicDeexcitation,
                    AtomicDataType::switchSpontaneousDeexcitation,
                    AtomicDataType::switchAutonomousIonization,
                    AtomicDataType::switchElectronicIonization,
                    AtomicDataType::switchFieldIonization>;
            PMACC_LOCKSTEP_KERNEL(ChooseTransitionTypeKernel())
                .config(mapper.getGridDim(), ions)(
                    mapper,
                    rngFactoryFloat,
                    localTimeStepField.getDeviceDataBox(),
                    localTimeRemainingField.getDeviceDataBox(),
                    localRateCacheField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
