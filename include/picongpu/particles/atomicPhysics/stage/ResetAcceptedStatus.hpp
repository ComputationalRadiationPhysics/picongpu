/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file resetAcceptedStatus sub-stage of atomicPhysics

#pragma once

#include "picongpu/particles/atomicPhysics/kernel/ResetAcceptedStatus.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomic physics sub-stage resetting the macro-ion attribute accepted to false
     *
     * @attention will break an in progress atomicPhysics step, only call at the start or
     *  end of the atomicPhysics step
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct ResetAcceptedStatus
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            PMACC_LOCKSTEP_KERNEL(picongpu::particles::atomicPhysics::kernel::ResetAcceptedStatusKernel())
                .config(
                    mapper.getGridDim(),
                    ions)(mapper, timeRemainingField.getDeviceDataBox(), ions.getDeviceParticlesBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
