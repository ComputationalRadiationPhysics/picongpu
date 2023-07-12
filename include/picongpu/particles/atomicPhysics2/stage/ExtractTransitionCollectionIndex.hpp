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

//! @file deduceTransitionCollectionIndex sub-stage of atomicPhysics

#pragma once

#include "picongpu/particles/atomicPhysics2/kernel/ExtractTransitionCollectionIndex_Autonomous.kernel"
#include "picongpu/particles/atomicPhysics2/kernel/ExtractTransitionCollectionIndex_BoundBound.kernel"
#include "picongpu/particles/atomicPhysics2/kernel/ExtractTransitionCollectionIndex_BoundFree.kernel"
#include "picongpu/particles/atomicPhysics2/kernel/ExtractTransitionCollectionIndex_NoChange.kernel"

namespace picongpu::particles::atomicPhysics2::stage
{
    /** @class atomic physics sub-stage extracting the transitionCollection index from
     *      the previously chosen global transition for he current atomic state
     *      for each macro-ion of the given species
     *
     * @attention assumes that the the choose transition kernel has been completed already
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct ExtractTransitionCollectionIndex
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

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            RngFactoryInt rngFactory = RngFactoryInt{currentStep};

            // no-change transition
            PMACC_LOCKSTEP_KERNEL(
                picongpu::particles::atomicPhysics2::kernel::ExtractTransitionCollectionIndexKernel_NoChange(),
                workerCfg)
            (mapper.getGridDim())(mapper, ions.getDeviceParticlesBox());

            if constexpr(
                AtomicDataType::switchElectronicExcitation || AtomicDataType::switchElectronicDeexcitation
                || AtomicDataType::switchSpontaneousDeexcitation)
            {
                // bound-bound transitions
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::ExtractTransitionCollectionIndexKernel_BoundBound<
                        SpeciesConfigNumberType,
                        picongpu::atomicPhysics2::ElectronHistogram::numberBins,
                        AtomicDataType::switchElectronicExcitation,
                        AtomicDataType::switchElectronicDeexcitation,
                        AtomicDataType::switchSpontaneousDeexcitation,
                        AtomicDataType::switchElectronicIonization,
                        AtomicDataType::switchAutonomousIonization,
                        AtomicDataType::switchFieldIonization>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactory,
                    ions.getDeviceParticlesBox(),
                    atomicData.template getChargeStateOrgaDataBox<false>(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundBoundNumberTransitionsDataBox<false>(),
                    atomicData.template getBoundBoundStartIndexBlockDataBox<false>());
            }

            if constexpr(AtomicDataType::switchElectronicIonization || AtomicDataType::switchFieldIonization)
            {
                // bound-free transitions
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::ExtractTransitionCollectionIndexKernel_BoundFree<
                        SpeciesConfigNumberType,
                        picongpu::atomicPhysics2::ElectronHistogram::numberBins,
                        AtomicDataType::switchElectronicExcitation,
                        AtomicDataType::switchElectronicDeexcitation,
                        AtomicDataType::switchSpontaneousDeexcitation,
                        AtomicDataType::switchElectronicIonization,
                        AtomicDataType::switchAutonomousIonization,
                        AtomicDataType::switchFieldIonization>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactory,
                    ions.getDeviceParticlesBox(),
                    atomicData.template getChargeStateOrgaDataBox<false>(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundFreeNumberTransitionsDataBox<false>(),
                    atomicData.template getBoundFreeStartIndexBlockDataBox<false>());
            }

            if constexpr(AtomicDataType::switchAutonomousIonization)
            {
                // autonomous transitions
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::ExtractTransitionCollectionIndexKernel_Autonomous<
                        SpeciesConfigNumberType,
                        AtomicDataType::switchElectronicExcitation,
                        AtomicDataType::switchElectronicDeexcitation,
                        AtomicDataType::switchSpontaneousDeexcitation,
                        AtomicDataType::switchElectronicIonization,
                        AtomicDataType::switchAutonomousIonization,
                        AtomicDataType::switchFieldIonization>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    ions.getDeviceParticlesBox(),
                    atomicData.template getChargeStateOrgaDataBox<false>(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getAutonomousNumberTransitionsDataBox<false>(),
                    atomicData.template getAutonomousStartIndexBlockDataBox<false>());
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage
