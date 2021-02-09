/* Copyright 2015-2021 Heiko Burau
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

#include "picongpu/particles/creation/creation.kernel"
#include "picongpu/simulation_defines.hpp"
#include <pmacc/cuSTL/algorithm/kernel/Foreach.hpp>
#include <pmacc/cuSTL/cursor/MultiIndexCursor.hpp>
#include <pmacc/cuSTL/zone/SphericZone.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace creation
        {
            /** Calls the `createParticlesKernel` kernel to create new particles.
             *
             * @param sourceSpecies species from which new particles are created
             * @param targetSpecies species of the created particles
             * @param particleCreator functor that defines the particle creation
             * @param cellDesc mapping description
             *
             * `particleCreator` must define: `init()`, `numNewParticles()` and `operator()()`
             * \see `PhotonCreator.hpp` for a further description.
             */
            template<
                typename T_SourceSpecies,
                typename T_TargetSpecies,
                typename T_ParticleCreator,
                typename T_CellDescription>
            void createParticlesFromSpecies(
                T_SourceSpecies& sourceSpecies,
                T_TargetSpecies& targetSpecies,
                T_ParticleCreator particleCreator,
                T_CellDescription cellDesc)
            {
                using SuperCellSize = typename MappingDesc::SuperCellSize;
                const pmacc::math::Int<simDim> coreBorderGuardSuperCells = cellDesc.getGridSuperCells();
                const pmacc::math::Int<simDim> guardSuperCells = cellDesc.getGuardingSuperCells();
                const pmacc::math::Int<simDim> coreBorderSuperCells = coreBorderGuardSuperCells - 2 * guardSuperCells;

                constexpr uint32_t numWorkers
                    = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                /* Functor holding the actual generic particle creation kernel */
                auto createParticlesKernel = make_CreateParticlesKernel<numWorkers>(
                    sourceSpecies.getDeviceParticlesBox(),
                    targetSpecies.getDeviceParticlesBox(),
                    particleCreator,
                    guardSuperCells);

                /* This zone represents the core+border area with guard offset in unit of cells */
                const zone::SphericZone<simDim> zone(
                    static_cast<pmacc::math::Size_t<simDim>>(coreBorderSuperCells * SuperCellSize::toRT()),
                    guardSuperCells * SuperCellSize::toRT());

                algorithm::kernel::ForeachLockstep<numWorkers, SuperCellSize> foreach;
                foreach(zone, createParticlesKernel, cursor::make_MultiIndexCursor<simDim>())
                    ;

                /* Make sure to leave no gaps in newly created frames */
                targetSpecies.fillAllGaps();
            }

        } // namespace creation
    } // namespace particles
} // namespace picongpu
