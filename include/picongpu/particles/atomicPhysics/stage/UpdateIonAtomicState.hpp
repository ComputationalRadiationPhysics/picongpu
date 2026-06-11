/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file record all ion transitions' delta energy

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics/kernel/UpdateIonAtomicState.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomicPhysics sub-stage updating atomic state according to accepted transitions, does nothing for ions which
     * did not accept a transition
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct UpdateIonAtomicState
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()([[maybe_unused]] picongpu::MappingDesc const mappingDesc) const
        {
            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;

            if constexpr(AtomicDataType::switchFieldIonization)
            {
                // full local domain, no guards
                pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
                pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                auto timeRemainingField
                    = dc.get<particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                        "TimeRemainingField");
                auto ions = dc.get<IonSpecies>(IonSpecies::FrameType::getName());
                auto atomicData = dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

                namespace s_enums = particles::atomicPhysics::enums;

                using UpdateIonAtomicState_fieldIonization
                    = picongpu::particles::atomicPhysics::kernel::UpdateIonAtomicStateKernel<
                        s_enums::ProcessClass::fieldIonization>;
                PMACC_LOCKSTEP_KERNEL(UpdateIonAtomicState_fieldIonization())
                    .config(mapper.getGridDim(), *ions)(
                        mapper,
                        timeRemainingField->getDeviceDataBox(),
                        ions->getDeviceParticlesBox(),
                        atomicData->template getAtomicStateDataDataBox<false>(),
                        atomicData->template getBoundFreeTransitionDataBox<
                            false,
                            s_enums::TransitionOrdering::byLowerState>());
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
