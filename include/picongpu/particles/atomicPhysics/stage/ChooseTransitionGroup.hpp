/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file ChooseTransitionGroup sub-stage of atomicPhysics
 *
 * randomly choose one transitionType for each macro-ion
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/kernel/ChooseTransitionGroup.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeStepField.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

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
    struct ChooseTransitionGroup
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

            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");
            auto& timeStepField
                = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::TimeStepField<picongpu::MappingDesc>>(
                    "TimeStepField");
            using RateCacheType = typename picongpu::particles::atomicPhysics::localHelperFields::
                RateCacheField<picongpu::MappingDesc, IonSpecies>::entryType;
            auto& rateCacheField = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                                               RateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_rateCacheField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());
            RngFactoryFloat rngFactoryFloat = RngFactoryFloat{currentStep};

            using ChooseTransitionGroupKernel =
                typename picongpu::particles::atomicPhysics::kernel::ChooseTransitionGroupKernel<
                    RateCacheType,
                    AtomicDataType::switchElectronicExcitation,
                    AtomicDataType::switchElectronicDeexcitation,
                    AtomicDataType::switchSpontaneousDeexcitation,
                    AtomicDataType::switchAutonomousIonization,
                    AtomicDataType::switchElectronicIonization,
                    AtomicDataType::switchFieldIonization>;
            PMACC_LOCKSTEP_KERNEL(ChooseTransitionGroupKernel())
                .config(mapper.getGridDim(), ions)(
                    mapper,
                    rngFactoryFloat,
                    timeStepField.getDeviceDataBox(),
                    timeRemainingField.getDeviceDataBox(),
                    rateCacheField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
