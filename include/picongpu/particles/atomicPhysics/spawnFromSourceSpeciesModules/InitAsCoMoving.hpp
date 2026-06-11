/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
