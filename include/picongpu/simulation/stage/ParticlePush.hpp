/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
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

#include "picongpu/particles/ParticlesFunctors.hpp"

#include <pmacc/eventSystem/Manager.hpp>

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            //! Functor for the stage of the PIC loop performing particle push
            struct ParticlePush
            {
                /** Push all particle species
                 *
                 * @param step index of time iteration
                 * @param[out] commEvent particle communication event
                 */
                void operator()(uint32_t const step, pmacc::EventTask& commEvent) const
                {
                    pmacc::EventTask initEvent = __getTransactionEvent();
                    pmacc::EventTask updateEvent;
                    particles::PushAllSpecies pushAllSpecies;
                    pushAllSpecies(step, initEvent, updateEvent, commEvent);
                    __setTransactionEvent(updateEvent);
                }
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
