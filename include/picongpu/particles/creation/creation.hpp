/*
 * SPDX-FileCopyrightText: Heiko Burau
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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

                DataConnector& dc = Environment<>::get().DataConnector();
                auto idProvider = dc.get<IdProvider>("globalId");

                PMACC_LOCKSTEP_KERNEL(CreateParticlesKernel{})
                    .config(mapper.getGridDim(), sourceSpecies)(
                        particleCreator,
                        sourceSpecies.getDeviceParticlesBox(),
                        targetSpecies.getDeviceParticlesBox(),
                        idProvider->getDeviceGenerator(),
                        mapper);

                /* Make sure to leave no gaps in newly created frames */
                targetSpecies.fillAllGaps();
            }

        } // namespace creation
    } // namespace particles
} // namespace picongpu
