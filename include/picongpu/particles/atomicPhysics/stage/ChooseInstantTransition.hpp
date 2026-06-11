/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file ChooseInstantTransition sub-stage of the atomicPhyiscs stage
 *
 * Selects one transition for all ions with a state loss rate above the instantaneous transition limit
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicData.hpp"
#include "picongpu/particles/atomicPhysics/kernel/ChooseInstantNonReversibleTransition.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/FoundUnboundIonField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetNumberAtomicStates.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    namespace s_enums = picongpu::particles::atomicPhysics::enums;

    /** ChooseInstantTransition atomic physics sub-stage
     *
     * @tparam T_IonSpecies ion species type
     *
     * @todo implement reversible instant transitions, Brian Marre, 2024
     *
     * @attention assumes that the flag accepted of all macro ions of T_IonSpecies has been reset before calling this
     * stage
     * @attention assumes that the foundUnboundIonField has been reset before calling this stage
     */
    template<typename T_IonSpecies>
    struct ChooseInstantTransition
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        // ionization potential depression model to use
        using IPDModel = picongpu::atomicPhysics::IPDModel;

        using DistributionFloat = pmacc::random::distributions::Uniform<float_X>;
        using RngFactoryFloat = particles::functor::misc::Rng<DistributionFloat>;

        //! call of kernel for every superCell
        HINLINE void operator()(
            [[maybe_unused]] picongpu::MappingDesc const mappingDesc,
            [[maybe_unused]] uint32_t const currentStep) const
        {
            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            if constexpr(AtomicDataType::switchFieldIonization)
            {
                // full local domain, no guards
                pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                auto timeRemainingField
                    = dc.get<particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                        "TimeRemainingField");
                auto foundUnboundIonField
                    = dc.get<particles::atomicPhysics::localHelperFields::FoundUnboundIonField<picongpu::MappingDesc>>(
                        "FoundUnboundIonField");
                auto eField = dc.get<FieldE>(FieldE::getName());
                auto ions = dc.get<IonSpecies>(IonSpecies::FrameType::getName());
                RngFactoryFloat rngFactoryFloat = RngFactoryFloat{currentStep};

                constexpr uint32_t numberAtomicStatesOfSpecies
                    = picongpu::traits::GetNumberAtomicStates<IonSpecies>::value;
                using ChooseInstantNonReversibleTransition = kernel::ChooseInstantNonReversibleTransitionKernel<
                    IPDModel,
                    numberAtomicStatesOfSpecies,
                    AtomicDataType::ADKLaserPolarization>;

                /// @todo implement iteration to capture field strength decrease, Brian Marre, 2024
                auto atomicData = dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");
                IPDModel::template callKernelWithIPDInput<
                    ChooseInstantNonReversibleTransition,
                    IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    rngFactoryFloat,
                    atomicData->template getChargeStateDataDataBox<false>(),
                    atomicData->template getAtomicStateDataDataBox<false>(),
                    atomicData->template getBoundFreeStartIndexBlockDataBox<false>(),
                    atomicData->template getBoundFreeNumberTransitionsDataBox<false>(),
                    atomicData
                        ->template getBoundFreeTransitionDataBox<false, s_enums::TransitionOrdering::byLowerState>(),
                    timeRemainingField->getDeviceDataBox(),
                    foundUnboundIonField->getDeviceDataBox(),
                    eField->getDeviceDataBox(),
                    ions->getDeviceParticlesBox());
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
