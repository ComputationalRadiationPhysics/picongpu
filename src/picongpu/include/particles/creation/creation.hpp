/**
 * Copyright 2015 Marco Garten, Heiko Burau
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "particles/creation/creation.kernel"
#include "simulation_defines.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "cuSTL/cursor/MultiIndexCursor.hpp"
#include "types.h"

namespace picongpu
{

using namespace PMacc;

namespace particles
{
namespace creation
{

template<typename SourceSpecies, typename TargetSpecies, typename ParticleCreator,
         typename CellDescription>
void createParticles(SourceSpecies& sourceSpecies, TargetSpecies& targetSpecies,
                     ParticleCreator& particleCreator, CellDescription* cellDesc)
{
    typedef typename MappingDesc::SuperCellSize SuperCellSize;
    const PMacc::math::Int<simDim> coreBorderGuardSuperCells = cellDesc->getGridSuperCells();
    const uint32_t guardSuperCells = cellDesc->getGuardingSuperCells();
    const PMacc::math::Int<simDim> coreBorderSuperCells = coreBorderGuardSuperCells - 2*guardSuperCells;

    /* Functor holding the actual generic particle creation kernel */
    PMACC_AUTO(createParticlesKernel, make_CreateParticlesKernel(
        sourceSpecies.getDeviceParticlesBox(),
        targetSpecies.getDeviceParticlesBox(),
        particleCreator));

    /* This zone represents the core+border area with guard offset in unit of cells */
    const zone::SphericZone<simDim> zone(
        static_cast<PMacc::math::Size_t<simDim> >(coreBorderSuperCells * SuperCellSize::toRT()),
        guardSuperCells * SuperCellSize::toRT());

    algorithm::kernel::Foreach<SuperCellSize> foreach;
    foreach(zone, cursor::make_MultiIndexCursor<simDim>(), createParticlesKernel);

    /* fill the gaps in the created species' particle frames to ensure that only
     * the last frame is not completely filled but every other before is full
     */
    targetSpecies.fillAllGaps();
}

} // namespace creation
} // namespace particles
} // namespace picongpu
