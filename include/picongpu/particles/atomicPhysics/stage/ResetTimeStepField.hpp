/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file resetTimeStepField sub-stage of atomicPhysics

#pragma once

#include "picongpu/particles/atomicPhysics/kernel/ResetTimeStepField.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeStepField.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** @class atomic physics sub-stage for resetting local atomicPhysics time step to
     *      current time remaining
     * @tparam T_numberAtomicPhysicsIonSpecies specialization template parameter used to prevent compilation of all
     *  atomicPhysics kernels if no atomic physics species is present.
     */
    template<uint32_t T_numberAtomicPhysicsIonSpecies>
    struct ResetTimeStepField
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

            // macro for call of kernel
            PMACC_LOCKSTEP_KERNEL(
                picongpu::particles::atomicPhysics::kernel::ResetTimeStepFieldKernel<
                    T_numberAtomicPhysicsIonSpecies>())
                .template config<1u>(mapper.getGridDim())(
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    timeStepField.getDeviceDataBox());
        }
    };

    template<>
    struct ResetTimeStepField<0u>
    {
        HINLINE void operator()([[maybe_unused]] picongpu::MappingDesc const mappingDesc) const
        {
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
