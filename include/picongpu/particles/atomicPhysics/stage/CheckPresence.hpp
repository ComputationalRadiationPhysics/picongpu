/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file checkForPresence sub-stage of atomicPhysics
 *
 * record all atomic states present in a superCell as present in the local rate Cache
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/kernel/CheckPresence.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"

#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <string>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomic physics sub-stage for recording all atomic states actually present in each superCell
     *
     * @tparam T_IonSpecies ion species type
     *
     * @attention assumes rateCacheField to have been reset before
     */
    template<typename T_IonSpecies>
    struct CheckPresence
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

            auto& timeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(
                "TimeRemainingField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());

            // pointers to memory, we will only work on device, no sync required
            //      pointer to rateCache
            auto& rateCacheField = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                                               RateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_rateCacheField");

            PMACC_LOCKSTEP_KERNEL(picongpu::particles::atomicPhysics::kernel::CheckPresenceKernel())
                .config(mapper.getGridDim(), ions)(
                    mapper,
                    timeRemainingField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox(),
                    rateCacheField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
