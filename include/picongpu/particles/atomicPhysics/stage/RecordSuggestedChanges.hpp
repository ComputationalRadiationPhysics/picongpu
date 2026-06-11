/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file record all accepted transition's suggested changes

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/kernel/RecordSuggestedChanges.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/FieldEnergyUseCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomicPhysics sub-stage recording for every accepted transition shared physics
     *  resource usage
     *
     * for example the histogram in weight usage of a collisional ionization,
     *  but not the ionization macro electron spawn, since that is not a shared resource.
     *
     * @attention assumes that the ChooseTransition, ExtractTransitionCollectionIndex
     *  and AcceptTransitionTest stages have been executed previously in the current
     *  atomicPhysics time step.
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct RecordSuggestedChanges
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

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;
            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");
            auto& electronHistogramField
                = *dc.get<particles::atomicPhysics::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_HistogramField");
            auto& fieldEnergyUseCacheField = *dc.get<
                particles::atomicPhysics::localHelperFields::FieldEnergyUseCacheField<picongpu::MappingDesc>>(
                "FieldEnergyUseCacheField");
            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            using IPDModel = picongpu::atomicPhysics::IPDModel;

            constexpr bool atLeastOneElectronicCollisionalChannelActive
                = AtomicDataType::switchElectronicExcitation || AtomicDataType::switchElectronicDeexcitation
                  || AtomicDataType::switchElectronicIonization;
            constexpr bool fieldIonizationActive = AtomicDataType::switchFieldIonization;

            if constexpr(atLeastOneElectronicCollisionalChannelActive || fieldIonizationActive)
            {
                IPDModel::template callKernelWithIPDInput<
                    particles::atomicPhysics::kernel::RecordSuggestedChangesKernel<
                        IPDModel,
                        atLeastOneElectronicCollisionalChannelActive,
                        fieldIonizationActive>,
                    IonSpecies::FrameType::frameSize>(
                    dc,
                    mapper,
                    atomicData.template getChargeStateDataDataBox<false>(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundFreeTransitionDataBox<
                        false,
                        picongpu::particles::atomicPhysics::enums::TransitionOrdering::byLowerState>(),
                    timeRemainingField.getDeviceDataBox(),
                    electronHistogramField.getDeviceDataBox(),
                    fieldEnergyUseCacheField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox());
            }

            /// @todo implement photonic collisional interactions, Brian Marre, 2023
        }
    };

} // namespace picongpu::particles::atomicPhysics::stage
