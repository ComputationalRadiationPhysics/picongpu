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

/** @file fill in local rate cache sub-stage of atomicPhysics for process class bound-bound
 *
 * implements filling upward bound-bound transitions' rates into the local rate cache,
 *  local rate cache accumulates by addition over all possible transitions.
 *
 * Used for atomicPhysics time step length calculation and as cache for local no-change
 *  transition rates.
 */

#pragma once

#include "picongpu/defines.hpp"
// need atomicPhysics.param, type of histogram

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicData.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/kernel/FillLocalRateCache_Autonomous.kernel"
#include "picongpu/particles/atomicPhysics/kernel/FillLocalRateCache_BoundBound.kernel"
#include "picongpu/particles/atomicPhysics/kernel/FillLocalRateCache_BoundFree.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalRateCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetNumberAtomicStates.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/lockstep/ForEach.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    namespace enums = picongpu::particles::atomicPhysics::enums;

    /** @class atomic physics sub-stage for filling transitions rates of one ion species
     *   into local rate caches in local domain
     *
     * @tparam T_IonSpecies ion species type
     *
     * @todo write unit test for this stage, Brian Marre , 2023
     */
    template<typename T_IonSpecies>
    struct FillLocalRateCache
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        // ionization potential depression model to use
        using IPDModel = picongpu::atomicPhysics::IPDModel;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localTimeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                "LocalTimeRemainingField");

            auto& localRateCacheField = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                                                    LocalRateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_localRateCacheField");

            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;
            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            constexpr uint8_t n_max = AtomicDataType::ConfigNumber::numberLevels;
            constexpr uint32_t numberAtomicStatesOfSpecies
                = picongpu::traits::GetNumberAtomicStates<IonSpecies>::value;
            constexpr uint32_t numberBins = picongpu::atomicPhysics::ElectronHistogram::numberBins;

            // filling local rate cache
            //    upward bound-bound transition rates
            if constexpr(AtomicDataType::switchElectronicExcitation)
            {
                using FillLocalRateCacheUpWardBoundBound = kernel::FillLocalRateCacheKernel_BoundBound<
                    n_max,
                    numberAtomicStatesOfSpecies,
                    numberBins,
                    AtomicDataType::switchElectronicExcitation,
                    AtomicDataType::switchElectronicDeexcitation,
                    AtomicDataType::switchSpontaneousDeexcitation,
                    enums::TransitionOrdering::byLowerState>;

                PMACC_LOCKSTEP_KERNEL(FillLocalRateCacheUpWardBoundBound())
                    .template config<IonSpecies::FrameType::frameSize>(mapper.getGridDim())(
                        mapper,
                        localTimeRemainingField.getDeviceDataBox(),
                        localRateCacheField.getDeviceDataBox(),
                        localElectronHistogramField.getDeviceDataBox(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData.template getBoundBoundStartIndexBlockDataBox<false>(),
                        atomicData.template getBoundBoundNumberTransitionsDataBox<false>(),
                        atomicData.template getBoundBoundTransitionDataBox<
                            false,
                            enums::TransitionOrdering::byLowerState>());
            }

            //    downward bound-bound transition rates
            if constexpr(AtomicDataType::switchElectronicDeexcitation || AtomicDataType::switchSpontaneousDeexcitation)
            {
                using FillLocalRateCacheDownWardBoundBound = kernel::FillLocalRateCacheKernel_BoundBound<
                    n_max,
                    numberAtomicStatesOfSpecies,
                    numberBins,
                    AtomicDataType::switchElectronicExcitation,
                    AtomicDataType::switchElectronicDeexcitation,
                    AtomicDataType::switchSpontaneousDeexcitation,
                    enums::TransitionOrdering::byUpperState>;

                PMACC_LOCKSTEP_KERNEL(FillLocalRateCacheDownWardBoundBound())
                    .template config<IonSpecies::FrameType::frameSize>(mapper.getGridDim())(
                        mapper,
                        localTimeRemainingField.getDeviceDataBox(),
                        localRateCacheField.getDeviceDataBox(),
                        localElectronHistogramField.getDeviceDataBox(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData.template getBoundBoundStartIndexBlockDataBox<false>(),
                        atomicData.template getBoundBoundNumberTransitionsDataBox<false>(),
                        atomicData.template getBoundBoundTransitionDataBox<
                            false,
                            enums::TransitionOrdering::byUpperState>());
            }

            //    upward bound-free transition rates, both collisional and field
            if constexpr(AtomicDataType::switchElectronicIonization)
            {
                auto eField = dc.get<FieldE>(FieldE::getName());

                using FillLocalRateCacheUpWardBoundFree = kernel::FillLocalRateCacheKernel_BoundFree<
                    IPDModel,
                    AtomicDataType::ADKLaserPolarization,
                    n_max,
                    numberAtomicStatesOfSpecies,
                    numberBins,
                    AtomicDataType::switchElectronicIonization,
                    AtomicDataType::switchFieldIonization,
                    enums::TransitionOrdering::byLowerState>;

                IPDModel::template callKernelWithIPDInput<
                    FillLocalRateCacheUpWardBoundFree,
                    IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    localRateCacheField.getDeviceDataBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    eField->getDeviceDataBox(),
                    atomicData.template getChargeStateDataDataBox<false>(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundFreeStartIndexBlockDataBox<false>(),
                    atomicData.template getBoundFreeNumberTransitionsDataBox<false>(),
                    atomicData
                        .template getBoundFreeTransitionDataBox<false, enums::TransitionOrdering::byLowerState>());
            }

            //    downward autonomous transition rates
            if constexpr(AtomicDataType::switchAutonomousIonization)
            {
                using FillLocalRateCacheAutonomous = kernel::FillLocalRateCacheKernel_Autonomous<
                    numberAtomicStatesOfSpecies,
                    AtomicDataType::switchAutonomousIonization,
                    enums::TransitionOrdering::byUpperState>;

                PMACC_LOCKSTEP_KERNEL(FillLocalRateCacheAutonomous())
                    .template config<IonSpecies::FrameType::frameSize>(mapper.getGridDim())(
                        mapper,
                        localTimeRemainingField.getDeviceDataBox(),
                        localRateCacheField.getDeviceDataBox(),
                        atomicData.template getAutonomousStartIndexBlockDataBox<false>(),
                        atomicData.template getAutonomousNumberTransitionsDataBox<false>(),
                        atomicData.template getAutonomousTransitionDataBox<
                            false,
                            enums::TransitionOrdering::byUpperState>());
            }
        }
    };

} // namespace picongpu::particles::atomicPhysics::stage
