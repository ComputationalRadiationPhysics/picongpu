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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics2/enums/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics2/kernel/RecordChanges.kernel"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalTimeRemainingField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2::stage
{
    /** atomicPhysics sub-stage recording deltaEnergy usage by all transitions
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct RecordDeltaEnergy
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg<IonSpecies::FrameType::frameSize>();

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            auto& localTimeRemainingField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<
                    picongpu::MappingDesc>>("LocalTimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics2::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            namespace enums = picongpu::particles::atomicPhysics2::enums;

            if constexpr(AtomicDataType::switchElectronicExcitation)
            {
                using RecordChanges_electronicExcitation
                    = picongpu::particles::atomicPhysics2 ::kernel::RecordChangesKernel<
                        enums::ProcessClass::electronicExcitation,
                        picongpu::atomicPhysics2::ElectronHistogram>;
                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicExcitation(), workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getBoundBoundTransitionDataBox<false, enums::TransitionOrdering::byLowerState>());
            }

            if constexpr(AtomicDataType::switchElectronicDeexcitation)
            {
                using RecordChanges_electronicDeexcitation
                    = picongpu::particles::atomicPhysics2::kernel::RecordChangesKernel<
                        enums::ProcessClass::electronicDeexcitation,
                        picongpu::atomicPhysics2::ElectronHistogram>;
                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicDeexcitation(), workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getBoundBoundTransitionDataBox<false, enums::TransitionOrdering::byUpperState>());
            }

            if constexpr(AtomicDataType::switchElectronicIonization)
            {
                using RecordChanges_electronicIonization
                    = picongpu::particles::atomicPhysics2::kernel::RecordChangesKernel<
                        enums::ProcessClass::electronicIonization,
                        picongpu::atomicPhysics2::ElectronHistogram>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicIonization(), workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getBoundFreeTransitionDataBox<false, enums::TransitionOrdering::byUpperState>(),
                    atomicData.template getChargeStateDataDataBox<false>());
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage
