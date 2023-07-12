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

//! @file record all ion transitions' delta energy

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics2/kernel/RecordChanges.kernel"
#include "picongpu/particles/atomicPhysics2/processClass/ProcessClass.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2::stage
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

        //! call of kernel for every superCell
        ALPAKA_FN_HOST void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics2::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            namespace procClass = picongpu::particles::atomicPhysics2::processClass;

            if constexpr(AtomicDataType::switchElectronicExcitation)
            {
                using RecordChanges_electronicExcitation
                    = picongpu::particles::atomicPhysics2 ::kernel::RecordChangesKernel<
                        procClass::ProcessClass::electronicExcitation,
                        picongpu::atomicPhysics2::ElectronHistogram>;
                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicExcitation(), workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundBoundTransitionDataBox<
                        false,
                        procClass::TransitionOrdering::byLowerState>());
            }

            if constexpr(AtomicDataType::switchElectronicDeexcitation)
            {
                using RecordChanges_electronicDeexcitation
                    = picongpu::particles::atomicPhysics2 ::kernel::RecordChangesKernel<
                        procClass::ProcessClass::electronicDeexcitation,
                        picongpu::atomicPhysics2::ElectronHistogram>;
                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicDeexcitation(), workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundBoundTransitionDataBox<
                        false,
                        procClass::TransitionOrdering::byUpperState>());
            }

            if constexpr(AtomicDataType::switchSpontaneousDeexcitation)
            {
                using RecordChanges_spontaneousDeexcitation
                    = picongpu::particles::atomicPhysics2::kernel ::RecordChangesKernel<
                        procClass::ProcessClass::spontaneousDeexcitation,
                        picongpu::atomicPhysics2::ElectronHistogram>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_spontaneousDeexcitation(), workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundBoundTransitionDataBox<
                        false,
                        procClass::TransitionOrdering::byUpperState>());
            }

            if constexpr(AtomicDataType::switchElectronicIonization)
            {
                using RecordChanges_electronicIonization
                    = picongpu::particles::atomicPhysics2::kernel ::RecordChangesKernel<
                        procClass::ProcessClass::electronicIonization,
                        picongpu::atomicPhysics2::ElectronHistogram>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_electronicIonization(), workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData
                        .template getBoundFreeTransitionDataBox<false, procClass::TransitionOrdering::byUpperState>(),
                    atomicData.template getChargeStateDataDataBox<false>());
            }

            // currently no bound-free based non collisional processes exist
            /// @todo implement field ionization, Brian Marre, 2023

            if constexpr(AtomicDataType::switchAutonomousIonization)
            {
                using RecordChanges_autonomousIonization
                    = picongpu::particles::atomicPhysics2::kernel ::RecordChangesKernel<
                        procClass::ProcessClass::autonomousIonization,
                        picongpu::atomicPhysics2::ElectronHistogram>;

                PMACC_LOCKSTEP_KERNEL(RecordChanges_autonomousIonization(), workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    ions.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getAutonomousTransitionDataBox<
                        false,
                        procClass::TransitionOrdering::byUpperState>());
            }
        }
    };

} // namespace picongpu::particles::atomicPhysics2::stage
