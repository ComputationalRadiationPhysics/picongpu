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

// need unit.param for normalisation and units, memory.param for SuperCellSize and dim.param for simDim
#include "picongpu/defines.hpp"
#include "picongpu/particles/creation/SpawnFromSourceSpeciesModuleInterfaces.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
{
    namespace s_interfaces = picongpu::particles::creation::moduleInterfaces;

    //! test for local time remaining <= 0 for superCell
    template<typename... T_KernelConfigOptions>
    struct SkipFinishedSuperCellsAtomicPhysics : public s_interfaces::SuperCellFilterFunctor<T_KernelConfigOptions...>
    {
        template<typename T_LocalTimeRemainingBox, typename... T_AdditionalStuff>
        HDINLINE static bool skipSuperCell(
            pmacc::DataSpace<picongpu::simDim> const,
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIndex,
            T_LocalTimeRemainingBox const timeRemainingBox,
            T_AdditionalStuff const...)
        {
            return (timeRemainingBox[superCellFieldIndex] <= 0._X);
        }
    };
} // namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
