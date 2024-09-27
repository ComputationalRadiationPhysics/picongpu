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

/** @file check for overSubscription of histogram bins and calculate rejectionProbability for each bin
 */

#pragma once

#include "picongpu/defines.hpp"
// need picongpu::atomicPhysics::ElectronHistogram from atomicPhysics.param

#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/kernel/CheckForOverSubscription.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalElectronHistogramOverSubscribedField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalRejectionProbabilityCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"

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

            auto& localTimeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                "LocalTimeRemainingField");

            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            auto& localElectronHistogramOverSubscribedField
                = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                              LocalElectronHistogramOverSubscribedField<picongpu::MappingDesc>>(
                    "LocalElectronHistogramOverSubscribedField");

            auto& localRejectionProbabilityCacheField
                = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::LocalRejectionProbabilityCacheField<
                    picongpu::MappingDesc>>("LocalRejectionProbabilityCacheField");

            // macro for call of kernel for every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(picongpu::particles::atomicPhysics::kernel::CheckForOverSubscriptionKernel<
                                      picongpu::atomicPhysics::ElectronHistogram,
                                      T_numberAtomicPhysicsIonSpecies>())
                .template config<picongpu::atomicPhysics::ElectronHistogram::numberBins>(mapper.getGridDim())(
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    localElectronHistogramOverSubscribedField.getDeviceDataBox(),
                    localRejectionProbabilityCacheField.getDeviceDataBox());

            /// @todo implement photon histogram, Brian Marre, 2023
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
