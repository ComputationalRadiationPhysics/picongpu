/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file dump rateCache to console, debug stage of atomicPhysics


#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/kernel/DumpRateCacheToConsole.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/RateCacheField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::stage
{
    /** @class atomicPhysics sub-stage dumping rateCache for one ion species to console,
     * calls the corresponding kernel per superCell
     *
     * is called once per time step for the entire local simulation volume by the atomicPhysicsStage
     */
    template<typename T_IonSpecies>
    struct DumpRateCacheToConsole
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_Species
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& rateCacheField = *dc.get<picongpu::particles::atomicPhysics::localHelperFields::
                                               RateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_rateCacheField");

            using DumpToConsole = picongpu::particles::atomicPhysics::kernel::DumpRateCacheToConsoleKernel;

            PMACC_LOCKSTEP_KERNEL(DumpToConsole())
                .template config<1u>(mapper.getGridDim())(mapper, rateCacheField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
