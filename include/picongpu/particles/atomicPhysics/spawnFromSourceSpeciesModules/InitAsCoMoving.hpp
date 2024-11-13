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
#include "picongpu/particles/atomicPhysics/initElectrons/CoMoving.hpp"
#include "picongpu/particles/creation/SpawnFromSourceSpeciesModuleInterfaces.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
{
    namespace s_interfaces = picongpu::particles::creation::moduleInterfaces;

    template<typename... T_KernelConfigOptions>
    struct InitAsCoMoving : public s_interfaces::ParticlePairUpdateFunctor<T_KernelConfigOptions...>
    {
        template<
            typename T_Worker,
            typename T_SourceParticle,
            typename T_ProductParticle,
            typename T_Number,
            typename T_KernelStateType,
            typename T_Index,
            typename... T_AdditionalData>
        HDINLINE static void update(
            T_Worker const& worker,
            T_SourceParticle& sourceParticle,
            T_ProductParticle& productParticle,
            IdGenerator& idGen,
            T_Number const,
            T_KernelStateType&,
            T_Index const,
            T_AdditionalData...)
        {
            particles::atomicPhysics::initElectrons::CoMoving::initElectron(
                worker,
                sourceParticle,
                productParticle,
                idGen);
        }
    };
} // namespace picongpu::particles::atomicPhysics::spawnFromSourceSpeciesModules
