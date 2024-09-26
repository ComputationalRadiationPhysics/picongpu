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

//! @file record all ion transitions' delta energy

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics/kernel/RecordChanges.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomicPhysics sub-stage recording deltaEnergy usage by all transitions
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

            auto& localTimeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                "LocalTimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            namespace enums = picongpu::particles::atomicPhysics::enums;

            if constexpr(AtomicDataType::switchElectronicExcitation)
            {
                using RecordChanges_electronicExcitation
                    = picongpu::particles::atomicPhysics ::kernel::RecordChangesKernel<
                        enums::ProcessClass::electronicExcitation,
                        picongpu::atomicPhysics::ElectronHistogram,
                        IPDModel>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicExcitation())
                    .config(mapper.getGridDim(), ions)(
                        mapper,
                        localTimeRemainingField.getDeviceDataBox(),
                        ions.getDeviceParticlesBox(),
                        localElectronHistogramField.getDeviceDataBox(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData.template getBoundBoundTransitionDataBox<
                            false,
                            enums::TransitionOrdering::byLowerState>());
            }

            if constexpr(AtomicDataType::switchElectronicDeexcitation)
            {
                using RecordChanges_electronicDeexcitation
                    = picongpu::particles::atomicPhysics::kernel::RecordChangesKernel<
                        enums::ProcessClass::electronicDeexcitation,
                        picongpu::atomicPhysics::ElectronHistogram,
                        IPDModel>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicDeexcitation())
                    .config(mapper.getGridDim(), ions)(
                        mapper,
                        localTimeRemainingField.getDeviceDataBox(),
                        ions.getDeviceParticlesBox(),
                        localElectronHistogramField.getDeviceDataBox(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData.template getBoundBoundTransitionDataBox<
                            false,
                            enums::TransitionOrdering::byUpperState>());
            }

            if constexpr(AtomicDataType::switchSpontaneousDeexcitation)
            {
                using RecordChanges_spontaneousDeexcitation
                    = picongpu::particles::atomicPhysics::kernel::RecordChangesKernel<
                        enums::ProcessClass::spontaneousDeexcitation,
                        picongpu::atomicPhysics::ElectronHistogram,
                        IPDModel>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_spontaneousDeexcitation())
                    .config(mapper.getGridDim(), ions)(
                        mapper,
                        localTimeRemainingField.getDeviceDataBox(),
                        ions.getDeviceParticlesBox(),
                        localElectronHistogramField.getDeviceDataBox(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData.template getBoundBoundTransitionDataBox<
                            false,
                            enums::TransitionOrdering::byUpperState>());
            }

            if constexpr(AtomicDataType::switchElectronicIonization)
            {
                using RecordChanges_electronicIonization
                    = picongpu::particles::atomicPhysics::kernel::RecordChangesKernel<
                        enums::ProcessClass::electronicIonization,
                        picongpu::atomicPhysics::ElectronHistogram,
                        IPDModel>;

                IPDModel::template callKernelWithIPDInput<
                    RecordChanges_electronicIonization,
                    IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getBoundFreeTransitionDataBox<false, enums::TransitionOrdering::byLowerState>(),
                    atomicData.template getChargeStateDataDataBox<false>());
            }

            // currently no bound-free based non-collisional processes exist
            /// @todo implement field ionization, Brian Marre, 2023

            if constexpr(AtomicDataType::switchAutonomousIonization)
            {
                using RecordChanges_autonomousIonization
                    = picongpu::particles::atomicPhysics::kernel::RecordChangesKernel<
                        enums::ProcessClass::autonomousIonization,
                        picongpu::atomicPhysics::ElectronHistogram,
                        IPDModel>;

                IPDModel::template callKernelWithIPDInput<
                    RecordChanges_autonomousIonization,
                    IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getAutonomousTransitionDataBox<false, enums::TransitionOrdering::byUpperState>());
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
