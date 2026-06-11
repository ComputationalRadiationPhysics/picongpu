/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

// need simDim from dimensions.param
#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/KernelIndexation.hpp"
#include "picongpu/particles/creation/SpawnFromSourceSpeciesModuleInterfaces.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
{
    namespace s_interfaces = picongpu::particles::creation::moduleInterfaces;

    template<typename... T_KernelConfigOptions>
    struct SuperCellFieldIndexFunctor : public s_interfaces::AdditionalDataIndexFunctor<T_KernelConfigOptions...>
    {
        template<typename T_AreaMapping>
        HDINLINE static pmacc::DataSpace<picongpu::simDim> getIndex(
            T_AreaMapping const areaMapping,
            pmacc::DataSpace<picongpu::simDim> const superCellIdx)
        {
            return particles::atomicPhysics::KernelIndexation::getSuperCellFieldIndexFromSuperCellIndex(
                areaMapping,
                superCellIdx);
        }
    };
} // namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
