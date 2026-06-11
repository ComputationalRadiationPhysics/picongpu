/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
