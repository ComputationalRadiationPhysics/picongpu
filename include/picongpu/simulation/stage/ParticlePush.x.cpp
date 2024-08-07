/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "picongpu/simulation/stage/ParticlePush.hpp"

#include "picongpu/particles/ParticlesFunctors.hpp"

#include <pmacc/eventSystem/Manager.hpp>

#include <cstdint>


namespace picongpu::simulation::stage
{
    void ParticlePush::operator()(uint32_t const step, pmacc::EventTask& commEvent) const
    {
        pmacc::EventTask initEvent = eventSystem::getTransactionEvent();
        pmacc::EventTask updateEvent;
        particles::PushAllSpecies pushAllSpecies;
        pushAllSpecies(step, initEvent, updateEvent, commEvent);
        eventSystem::setTransactionEvent(updateEvent);
    }
} // namespace picongpu::simulation::stage
