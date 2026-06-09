/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2023-2024 Brian Marre
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
