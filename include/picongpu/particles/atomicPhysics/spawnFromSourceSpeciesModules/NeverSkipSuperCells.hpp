/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

// need unit.param for normalisation and units, memory.param for SuperCellSize and dim.param for simDim
#include "picongpu/defines.hpp"
#include "picongpu/particles/creation/SpawnFromSourceSpeciesModuleInterfaces.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
{
    namespace s_interfaces = picongpu::particles::creation::moduleInterfaces;

    //! test for local time remaining <= 0 for superCell
    template<typename... T_KernelConfigOptions>
    struct NeverSkipSuperCells : public s_interfaces::SuperCellFilterFunctor<T_KernelConfigOptions...>
    {
        template<typename... T_AdditionalStuff>
        HDINLINE static bool skipSuperCell(
            pmacc::DataSpace<picongpu::simDim> const,
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIndex,
            T_AdditionalStuff const...)
        {
            return false;
        }
    };
} // namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
