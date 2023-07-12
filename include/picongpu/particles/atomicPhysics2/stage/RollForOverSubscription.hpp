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

#pragma once

#include "picongpu/particles/atomicPhysics2/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics2/kernel/RollForOverSubscription.kernel"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalElectronHistogramOverSubscribedField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalRejectionProbabilityCacheField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalTimeRemainingField.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::stage
{
    /** @class atomic physics sub-stage for rejection previously accepted transitions
     *      taking from oversubscribed bins
     *
     * For every macro-ion with an accepted transition, using weight from an over
     *  subscribed bin, try to reject it's transition once, with the bin's rejection
     *  probability stored in the superCell local cache.
     * The RejectionProbabilityCache is filled by the checkForOverSubscription stage.
     *
     * @attention assumes that the checkForOverSubscription stage has been called before
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct RollForOverSubscription
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        using DistributionFloat = pmacc::random::distributions::Uniform<float_X>;
        using RngFactoryFloat = particles::functor::misc::Rng<DistributionFloat>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg<IonSpecies::FrameType::frameSize>();

            auto& localTimeRemainingField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<
                    picongpu::MappingDesc>>("LocalTimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            auto& localRejectionProbabilityCacheField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalRejectionProbabilityCacheField<
                    picongpu::MappingDesc>>("LocalRejectionProbabilityCacheField");

            auto& localElectronHistogramOverSubscribedField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::
                              LocalElectronHistogramOverSubscribedField<picongpu::MappingDesc>>(
                    "LocalElectronHistogramOverSubscribedField");

            RngFactoryFloat rngFactory = RngFactoryFloat{currentStep};

            // macro for call of kernel for every superCell
            PMACC_LOCKSTEP_KERNEL(
                picongpu::particles::atomicPhysics2::kernel::RollForOverSubscriptionKernel<
                    picongpu::atomicPhysics2::ElectronHistogram>(),
                workerCfg)
            (mapper.getGridDim())(
                mapper,
                rngFactory,
                localTimeRemainingField.getDeviceDataBox(),
                localElectronHistogramOverSubscribedField.getDeviceDataBox(),
                ions.getDeviceParticlesBox(),
                localRejectionProbabilityCacheField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage
