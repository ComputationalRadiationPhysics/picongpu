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

#include "types.h"
#include "simulation_defines.hpp"
#include <boost/mpl/if.hpp>
#include "traits/HasFlag.hpp"
#include "fields/Fields.def"
#include "math/MapTuple.hpp"
#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>

#include "communication/AsyncCommunication.hpp"
#include "particles/creation/creation.kernel"

namespace picongpu
{

namespace particles
{

namespace creation
{

template<typename SourceSpecies, typename TargetSpecies, typename ParticleCreator,
         typename CellDescription>
void createParticles(SourceSpecies& sourceSpecies, TargetSpecies& targetSpecies,
                     ParticleCreator& particleCreator, CellDescription* cellDesc)
{
    /* 3-dim vector : number of threads to be started in every dimension */
    dim3 block( MappingDesc::SuperCellSize::toRT().toDim3() );

    /** kernelCreateParticles
     * \brief calls the particle creation kernel and handles that target particles are created correctly
     *        while cycling through the particle frames
     *
     * kernel call : instead of name<<<blocks, threads>>> (args, ...)
     * "blocks" will be calculated from "this->cellDescription" and "CORE + BORDER"
     * "threads" is calculated from the previously defined vector "block"
     */
    __picKernelArea( particles::creation::kernelCreateParticles, *cellDesc, CORE + BORDER )
        (block)
        ( sourceSpecies.getDeviceParticlesBox( ),
          targetSpecies.getDeviceParticlesBox( ),
          particleCreator
        );
    /* fill the gaps in the created species' particle frames to ensure that only
     * the last frame is not completely filled but every other before is full
     */
    targetSpecies.fillAllGaps();
}

} // namespace creation
} // namespace particles
} // namespace picongpu
