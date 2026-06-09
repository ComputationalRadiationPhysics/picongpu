/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

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

//! @file record all accepted transition's suggested changes

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics/kernel/RecordSuggestedFieldEnergyUse.kernel"
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
    struct RecordSuggestedFieldEnergyUse
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
            auto atomicData = dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            auto timeRemainingField = dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");
            auto fieldEnergyUseCacheField
                = dc.get<particles::atomicPhysics::localHelperFields::FieldEnergyUseCacheField<picongpu::MappingDesc>>(
                    "FieldEnergyUseCacheField");
            auto ions = dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            using IPDModel = picongpu::atomicPhysics::IPDModel;

            IPDModel::template callKernelWithIPDInput<
                particles::atomicPhysics::kernel::RecordSuggestedFieldEnergyUseKernel<IPDModel>,
                IonSpecies::FrameType::frameSize>(
                dc,
                mapper,
                atomicData->template getChargeStateDataDataBox<false>(),
                atomicData->template getAtomicStateDataDataBox<false>(),
                atomicData->template getBoundFreeTransitionDataBox<
                    false,
                    picongpu::particles::atomicPhysics::enums::TransitionOrdering::byLowerState>(),
                timeRemainingField->getDeviceDataBox(),
                fieldEnergyUseCacheField->getDeviceDataBox(),
                ions->getDeviceParticlesBox());
        }
    };

} // namespace picongpu::particles::atomicPhysics::stage
