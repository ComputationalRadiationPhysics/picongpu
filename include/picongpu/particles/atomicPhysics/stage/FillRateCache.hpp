/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file fill in rate cache sub-stage of atomicPhysics for process class bound-bound
 *
 * implements filling upward bound-bound transitions' rates into the rate cache,
 *  rate cache accumulates by addition over all possible transitions.
 *
 * Used for atomicPhysics time step length calculation and as cache for no-change transition rates.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicData.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/kernel/FillRateCache_Autonomous.kernel"
#include "picongpu/particles/atomicPhysics/kernel/FillRateCache_BoundBound.kernel"
#include "picongpu/particles/atomicPhysics/kernel/FillRateCache_BoundFree.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCacheField.hpp"
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
    namespace enums = picongpu::particles::atomicPhysics::enums;

    /** @class atomic physics sub-stage for filling transitions rates of one ion species
     *   into local rate caches in local domain
     *
     * @tparam T_IonSpecies ion species type
     *
     * @todo write unit test for this stage, Brian Marre , 2023
     */
    template<typename T_IonSpecies>
    struct FillRateCache
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

            auto timeRemainingField = dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");

            auto rateCacheField = dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                                             RateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_rateCacheField");

            auto electronHistogramField
                = dc.get<picongpu::particles::atomicPhysics::electronDistribution::
                             LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_HistogramField");

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;
            auto atomicData = dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            constexpr uint8_t n_max = AtomicDataType::ConfigNumber::numberLevels;
            constexpr uint32_t numberAtomicStatesOfSpecies
                = picongpu::traits::GetNumberAtomicStates<IonSpecies>::value;
            constexpr uint32_t numberBins = picongpu::atomicPhysics::ElectronHistogram::numberBins;

            // filling rate cache
            //    upward bound-bound transition rates
            if constexpr(AtomicDataType::switchElectronicExcitation)
            {
                using FillRateCacheUpWardBoundBound = kernel::FillRateCacheKernel_BoundBound<
                    n_max,
                    numberAtomicStatesOfSpecies,
                    numberBins,
                    AtomicDataType::switchElectronicExcitation,
                    AtomicDataType::switchElectronicDeexcitation,
                    AtomicDataType::switchSpontaneousDeexcitation,
                    enums::TransitionOrdering::byLowerState>;

                PMACC_LOCKSTEP_KERNEL(FillRateCacheUpWardBoundBound())
                    .template config<IonSpecies::FrameType::frameSize>(mapper.getGridDim())(
                        mapper,
                        timeRemainingField->getDeviceDataBox(),
                        rateCacheField->getDeviceDataBox(),
                        electronHistogramField->getDeviceDataBox(),
                        atomicData->template getAtomicStateDataDataBox<false>(),
                        atomicData->template getBoundBoundStartIndexBlockDataBox<false>(),
                        atomicData->template getBoundBoundNumberTransitionsDataBox<false>(),
                        atomicData->template getBoundBoundTransitionDataBox<
                            false,
                            enums::TransitionOrdering::byLowerState>());
            }

            //    downward bound-bound transition rates
            if constexpr(AtomicDataType::switchElectronicDeexcitation || AtomicDataType::switchSpontaneousDeexcitation)
            {
                using FillRateCacheDownWardBoundBound = kernel::FillRateCacheKernel_BoundBound<
                    n_max,
                    numberAtomicStatesOfSpecies,
                    numberBins,
                    AtomicDataType::switchElectronicExcitation,
                    AtomicDataType::switchElectronicDeexcitation,
                    AtomicDataType::switchSpontaneousDeexcitation,
                    enums::TransitionOrdering::byUpperState>;

                PMACC_LOCKSTEP_KERNEL(FillRateCacheDownWardBoundBound())
                    .template config<IonSpecies::FrameType::frameSize>(mapper.getGridDim())(
                        mapper,
                        timeRemainingField->getDeviceDataBox(),
                        rateCacheField->getDeviceDataBox(),
                        electronHistogramField->getDeviceDataBox(),
                        atomicData->template getAtomicStateDataDataBox<false>(),
                        atomicData->template getBoundBoundStartIndexBlockDataBox<false>(),
                        atomicData->template getBoundBoundNumberTransitionsDataBox<false>(),
                        atomicData->template getBoundBoundTransitionDataBox<
                            false,
                            enums::TransitionOrdering::byUpperState>());
            }

            //    upward bound-free transition rates, both collisional and field
            if constexpr(AtomicDataType::switchElectronicIonization || AtomicDataType::switchFieldIonization)
            {
                auto& eField = *dc.get<FieldE>(FieldE::getName());

                using FillRateCacheUpWardBoundFree = kernel::FillRateCacheKernel_BoundFree<
                    IPDModel,
                    AtomicDataType::ADKLaserPolarization,
                    n_max,
                    numberAtomicStatesOfSpecies,
                    numberBins,
                    AtomicDataType::switchElectronicIonization,
                    AtomicDataType::switchFieldIonization,
                    enums::TransitionOrdering::byLowerState>;

                IPDModel::template callKernelWithIPDInput<
                    FillRateCacheUpWardBoundFree,
                    IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    timeRemainingField->getDeviceDataBox(),
                    rateCacheField->getDeviceDataBox(),
                    electronHistogramField->getDeviceDataBox(),
                    eField.getDeviceDataBox(),
                    atomicData->template getChargeStateDataDataBox<false>(),
                    atomicData->template getAtomicStateDataDataBox<false>(),
                    atomicData->template getBoundFreeStartIndexBlockDataBox<false>(),
                    atomicData->template getBoundFreeNumberTransitionsDataBox<false>(),
                    atomicData
                        ->template getBoundFreeTransitionDataBox<false, enums::TransitionOrdering::byLowerState>());
            }

            //    downward autonomous transition rates
            if constexpr(AtomicDataType::switchAutonomousIonization)
            {
                using FillRateCacheAutonomous = kernel::FillRateCacheKernel_Autonomous<
                    numberAtomicStatesOfSpecies,
                    AtomicDataType::switchAutonomousIonization,
                    enums::TransitionOrdering::byUpperState>;

                PMACC_LOCKSTEP_KERNEL(FillRateCacheAutonomous())
                    .template config<IonSpecies::FrameType::frameSize>(mapper.getGridDim())(
                        mapper,
                        timeRemainingField->getDeviceDataBox(),
                        rateCacheField->getDeviceDataBox(),
                        atomicData->template getAutonomousStartIndexBlockDataBox<false>(),
                        atomicData->template getAutonomousNumberTransitionsDataBox<false>(),
                        atomicData->template getAutonomousTransitionDataBox<
                            false,
                            enums::TransitionOrdering::byUpperState>());
            }
        }
    };

} // namespace picongpu::particles::atomicPhysics::stage
