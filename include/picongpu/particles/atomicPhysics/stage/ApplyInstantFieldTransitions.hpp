/* Copyright 2024 Brian Marre
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicData.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/kernel/ApplyInstantFieldTransitions.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"
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

    /** accept field ionization processes with a rate above the time resolution limit instantaneously
     *
     * This kernel implements a stop gap fix for field ionization lacking energy conservation.
     *
     * In contrast to collisional transitions field ionization rates do not reduce over consecutive atomicPhysics
     *  subSteps due to exhaustion of the local field, since the local fields are not updated by field ionization
     *  transitions yet.
     * This in addition with the very high field ionization rates at laser peak field strength reduces the
     * atomicPhysics sub step time step length below the numerical resolution limit, causing a very high number of sub
     * steps and the accompanying performance issues.
     *
     * To avoid this we apply field ionization transitions instantaneously to atomic states with field ionization rates
     *  above the user set time step resolution limit before we calculate the time step length.
     * Thereby letting very fast transitions occur "instantaneously" in the sub-stepping and removing them from the
     * rate solver.
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct ApplyInstantFieldTransitions
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        // ionization potential depression model to use
        using IPDModel = picongpu::atomicPhysics::IPDModel;

        using DistributionFloat = pmacc::random::distributions::Uniform<float_X>;
        using RngFactoryFloat = particles::functor::misc::Rng<DistributionFloat>;

        //! call of kernel for every superCell
        HINLINE void operator()([[maybe_unused]] picongpu::MappingDesc const mappingDesc, uint32_t const currentStep)
            const
        {
            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            if constexpr(AtomicDataType::switchFieldIonization)
            {
                // full local domain, no guards
                pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                constexpr uint32_t numberAtomicStatesOfSpecies
                    = picongpu::traits::GetNumberAtomicStates<IonSpecies>::value;

                auto& localTimeRemainingField
                    = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<
                        picongpu::MappingDesc>>("LocalTimeRemainingField");
                auto& localFoundUnboundIonField
                    = *dc.get<atomicPhysics::localHelperFields::LocalFoundUnboundIonField<picongpu::MappingDesc>>(
                        "LocalFoundUnboundIonField");

                using ApplyInstantFieldTransitions = kernel::ApplyInstantFieldTransitionsKernel<
                    IPDModel,
                    AtomicDataType::ADKLaserPolarization,
                    FieldE,
                    numberAtomicStatesOfSpecies,
                    enums::TransitionOrdering::byLowerState>;

                RngFactoryFloat rngFactoryFloat = RngFactoryFloat{currentStep};
                auto eField = dc.get<FieldE>(FieldE::getName());
                auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");
                auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

                IPDModel::
                    template callKernelWithIPDInput<ApplyInstantFieldTransitions, IonSpecies::FrameType::frameSize>(
                        dc,
                        mapper,
                        rngFactoryFloat,
                        localTimeRemainingField.getDeviceDataBox(),
                        localFoundUnboundIonField.getDeviceDataBox(),
                        eField->getDeviceDataBox(),
                        atomicData.template getChargeStateDataDataBox<false>(),
                        atomicData.template getAtomicStateDataDataBox<false>(),
                        atomicData.template getBoundFreeStartIndexBlockDataBox<false>(),
                        atomicData.template getBoundFreeNumberTransitionsDataBox<false>(),
                        atomicData
                            .template getBoundFreeTransitionDataBox<false, enums::TransitionOrdering::byLowerState>(),
                        ions.getDeviceParticlesBox());
            }
        }
    };

} // namespace picongpu::particles::atomicPhysics::stage
