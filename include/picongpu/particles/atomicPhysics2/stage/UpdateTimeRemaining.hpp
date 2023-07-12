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

//! @file UpdateTimeRemaining sub-stage of atomicPhysics

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/kernel/UpdateTimeRemaining.kernel"

#include <string>

namespace picongpu::particles::atomicPhysics2::stage
{
    /** atomic physics sub-stage for reducing the local time remaining by the local
     *  atomicPhysics time step
     */
    struct UpdateTimeRemaining
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg<1u>();

            // pointers to memory, we will only work on device, no sync required
            //      pointer to localTimeRemainingField
            auto& localTimeRemainingField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<
                    picongpu::MappingDesc>>("LocalTimeRemainingField");
            //      pointer to localTimeStepFieldField
            auto& localTimeStepField = *dc.get<
                picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeStepField<picongpu::MappingDesc>>(
                "LocalTimeStepField");

            // macro for kernel call
            PMACC_LOCKSTEP_KERNEL(picongpu::particles::atomicPhysics2::kernel::UpdateTimeRemainingKernel(), workerCfg)
            (mapper.getGridDim())(
                mapper,
                localTimeRemainingField.getDeviceDataBox(),
                localTimeStepField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage
