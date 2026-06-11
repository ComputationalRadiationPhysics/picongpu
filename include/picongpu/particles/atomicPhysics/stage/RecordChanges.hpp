/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file record all ion transitions' delta energy

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics/kernel/RecordChanges.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomicPhysics sub-stage updating atomic state of ions and recording electron histogram energy usage of
     *  transitions
     *
     * @attention assumes that all ions accepted a transition, no check for acceptance outside of a debug compile.
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct RecordChanges
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        // ionization potential depression model to use for energy calculation
        using IPDModel = picongpu::atomicPhysics::IPDModel;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            auto& electronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_HistogramField");

            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            namespace s_enums = picongpu::particles::atomicPhysics::enums;

            if constexpr(AtomicDataType::switchElectronicExcitation)
            {
                using RecordChanges_electronicExcitation = picongpu::particles::atomicPhysics ::kernel::
                    RecordChangesKernel<s_enums::ProcessClass::electronicExcitation, IPDModel>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicExcitation())
                    .config(mapper.getGridDim(), ions)(
                        mapper,
                        timeRemainingField.getDeviceDataBox(),
                        ions.getDeviceParticlesBox(),
                        electronHistogramField.getDeviceDataBox(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData.template getBoundBoundTransitionDataBox<
                            false,
                            s_enums::TransitionOrdering::byLowerState>());
            }

            if constexpr(AtomicDataType::switchElectronicDeexcitation)
            {
                using RecordChanges_electronicDeexcitation = picongpu::particles::atomicPhysics::kernel::
                    RecordChangesKernel<s_enums::ProcessClass::electronicDeexcitation, IPDModel>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicDeexcitation())
                    .config(mapper.getGridDim(), ions)(
                        mapper,
                        timeRemainingField.getDeviceDataBox(),
                        ions.getDeviceParticlesBox(),
                        electronHistogramField.getDeviceDataBox(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData.template getBoundBoundTransitionDataBox<
                            false,
                            s_enums::TransitionOrdering::byUpperState>());
            }

            if constexpr(AtomicDataType::switchSpontaneousDeexcitation)
            {
                using RecordChanges_spontaneousDeexcitation = picongpu::particles::atomicPhysics::kernel::
                    RecordChangesKernel<s_enums::ProcessClass::spontaneousDeexcitation, IPDModel>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_spontaneousDeexcitation())
                    .config(mapper.getGridDim(), ions)(
                        mapper,
                        timeRemainingField.getDeviceDataBox(),
                        ions.getDeviceParticlesBox(),
                        electronHistogramField.getDeviceDataBox(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData.template getBoundBoundTransitionDataBox<
                            false,
                            s_enums::TransitionOrdering::byUpperState>());
            }

            if constexpr(AtomicDataType::switchElectronicIonization)
            {
                using RecordChanges_electronicIonization = picongpu::particles::atomicPhysics::kernel::
                    RecordChangesKernel<s_enums::ProcessClass::electronicIonization, IPDModel>;

                IPDModel::template callKernelWithIPDInput<
                    RecordChanges_electronicIonization,
                    IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    electronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getBoundFreeTransitionDataBox<false, s_enums::TransitionOrdering::byLowerState>(),
                    atomicData.template getChargeStateDataDataBox<false>());
            }

            if constexpr(AtomicDataType::switchFieldIonization)
            {
                using RecordChanges_fieldIonization = picongpu::particles::atomicPhysics::kernel::
                    RecordChangesKernel<s_enums::ProcessClass::fieldIonization, IPDModel>;

                IPDModel::template callKernelWithIPDInput<
                    RecordChanges_fieldIonization,
                    IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    electronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getBoundFreeTransitionDataBox<false, s_enums::TransitionOrdering::byLowerState>(),
                    atomicData.template getChargeStateDataDataBox<false>());
            }

            if constexpr(AtomicDataType::switchAutonomousIonization)
            {
                using RecordChanges_autonomousIonization = picongpu::particles::atomicPhysics::kernel::
                    RecordChangesKernel<s_enums::ProcessClass::autonomousIonization, IPDModel>;

                IPDModel::template callKernelWithIPDInput<
                    RecordChanges_autonomousIonization,
                    IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    electronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getAutonomousTransitionDataBox<false, s_enums::TransitionOrdering::byUpperState>());
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
