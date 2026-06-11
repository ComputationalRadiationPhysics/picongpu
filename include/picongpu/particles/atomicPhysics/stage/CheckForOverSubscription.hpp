/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file check for overSubscription of histogram bins and cells and calculate rejectionProbability for each*/

#pragma once

// need picongpu::atomicPhysics::ElectronHistogram from atomicPhysics.param
#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/kernel/CheckForOverSubscription.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/FieldEnergyUseCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RejectionProbabilityCacheField_Bin.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RejectionProbabilityCacheField_Cell.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/SharedResourcesOverSubscribedField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::stage
{
    /** CheckForAndRejectOversubscription atomic physics sub-stage
     *
     * check each histogram bin for deltaWeight > weight0, if yes mark bin as over subscribed
     *
     * @tparam T_numberAtomicPhysicsIonSpecies specialization template parameter used to prevent compilation of all
     *  atomicPhysics kernels if no atomic physics species is present.
     */
    template<uint32_t T_numberAtomicPhysicsIonSpecies>
    struct CheckForOverSubscription
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");

            auto& electronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_HistogramField");

            auto& sharedResourcesOverSubscribedField
                = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::SharedResourcesOverSubscribedField<
                    picongpu::MappingDesc>>("SharedResourcesOverSubscribedField");

            auto& rejectionProbabilityCacheField_Bin
                = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::RejectionProbabilityCacheField_Bin<
                    picongpu::MappingDesc>>("RejectionProbabilityCacheField_Bin");
            auto& rejectionProbabilityCacheField_Cell
                = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::RejectionProbabilityCacheField_Cell<
                    picongpu::MappingDesc>>("RejectionProbabilityCacheField_Cell");

            auto& eField = *dc.get<FieldE>(FieldE::getName());

            using FieldEnergyUseCacheField
                = picongpu::particles::atomicPhysics::localHelperFields::FieldEnergyUseCacheField<
                    picongpu::MappingDesc>;
            auto& fieldEnergyUseCacheField = *dc.get<FieldEnergyUseCacheField>("FieldEnergyUseCacheField");

            // macro for call of kernel for every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(
                picongpu::particles::atomicPhysics::kernel::CheckForOverSubscriptionKernel<
                    T_numberAtomicPhysicsIonSpecies>())
                .template config<picongpu::atomicPhysics::ElectronHistogram::numberBins>(mapper.getGridDim())(
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    electronHistogramField.getDeviceDataBox(),
                    eField.getDeviceDataBox(),
                    fieldEnergyUseCacheField.getDeviceDataBox(),
                    sharedResourcesOverSubscribedField.getDeviceDataBox(),
                    rejectionProbabilityCacheField_Bin.getDeviceDataBox(),
                    rejectionProbabilityCacheField_Cell.getDeviceDataBox());

            /// @todo implement photon histogram, Brian Marre, 2023
        }
    };

    //! specialization for no atomicPhysics ion species
    template<>
    struct CheckForOverSubscription<0u>
    {
        HINLINE void operator()([[maybe_unused]] picongpu::MappingDesc const mappingDesc) const
        {
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
