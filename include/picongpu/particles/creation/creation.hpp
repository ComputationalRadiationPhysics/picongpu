/* Copyright 2015-2023 Heiko Burau
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/creation/creation.kernel"

#include <pmacc/lockstep/lockstep.hpp>

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
                auto const mapper = makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(cellDesc);

                auto workerCfg = pmacc::lockstep::makeWorkerCfg<T_SourceSpecies::FrameType::frameSize>();
                PMACC_LOCKSTEP_KERNEL(CreateParticlesKernel{}, workerCfg)
                (mapper.getGridDim())(
                    particleCreator,
                    sourceSpecies.getDeviceParticlesBox(),
                    targetSpecies.getDeviceParticlesBox(),
                    mapper);

                /* Make sure to leave no gaps in newly created frames */
                targetSpecies.fillAllGaps();
            }

        } // namespace creation
    } // namespace particles
} // namespace picongpu
