/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file dump particle information to console, debug stage of atomicPhysics


#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/kernel/DumpAllIonsToConsole.kernel"
#include "picongpu/particles/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::debug::stage
{
    /** @class atomicPhysics sub-stage dumping all macro ion atomicPhysics data for a species to console
     * calls the corresponding kernel per superCell
     *
     * is called once per time step for the entire local simulation volume and for
     * every isElectron species by the atomicPhysics stage
     *
     * @tparam T_ElectronSpecies species for which to call the functor
     */
    template<typename T_Species, typename T_ParticleFilter>
    struct DumpAllIonsToConsole
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_Species
        using Species = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_Species>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            // pointer to memory, we will only work on device, no sync required
            // init pointer to macro particles
            auto& particles = *dc.get<Species>(Species::FrameType::getName());

            using DumpToConsole
                = picongpu::particles::atomicPhysics::debug::kernel::DumpAllIonsToConsoleKernel<T_ParticleFilter>;

            // macro for call of kernel on every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(DumpToConsole())
                .config(mapper.getGridDim(), particles)(mapper, particles.getDeviceParticlesBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::debug::stage
