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

//! @file UpdateTimeRemaining sub-stage of atomicPhysics

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/kernel/UpdateTimeRemaining.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeStepField.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomic physics sub-stage for reducing the local time remaining by the local
     *  atomicPhysics time step
     *
     * @tparam T_numberAtomicPhysicsIonSpecies only used to prevent compilation of atomicPhysics kernels if no atomic
     *  physics ion specie present
     */
    template<uint32_t T_numberAtomicPhysicsIonSpecies>
    struct UpdateTimeRemaining
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            // pointers to memory, we will only work on device, no sync required
            //      pointer to timeRemainingField
            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");
            //      pointer to timeStepFieldField
            auto& timeStepField
                = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::TimeStepField<picongpu::MappingDesc>>(
                    "TimeStepField");

            // macro for kernel call
            PMACC_LOCKSTEP_KERNEL(picongpu::particles::atomicPhysics::kernel::UpdateTimeRemainingKernel())
                .template config<1u>(mapper.getGridDim())(
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    timeStepField.getDeviceDataBox());
        }
    };

    //! specialization for no atomic Physics ion specie sin simulation
    template<>
    struct UpdateTimeRemaining<0u>
    {
        HINLINE void operator()([[maybe_unused]] picongpu::MappingDesc const mappingDesc) const
        {
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
