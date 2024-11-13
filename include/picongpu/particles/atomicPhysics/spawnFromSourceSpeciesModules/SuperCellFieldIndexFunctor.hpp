/* Copyright 2024 Brian Marre
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
