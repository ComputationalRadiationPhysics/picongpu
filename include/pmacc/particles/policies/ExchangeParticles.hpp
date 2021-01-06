/* Copyright 2015-2021 Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/Environment.hpp"

namespace pmacc
{
    namespace particles
    {
        namespace policies
        {
            /**
             * Policy for \see HandleGuardRegion that moves particles from guard cells to exchange buffers
             * and sends those to the correct neighbors
             */
            struct ExchangeParticles
            {
                template<class T_Particles>
                void handleOutgoing(T_Particles& par, int32_t direction) const
                {
                    Environment<>::get().ParticleFactory().createTaskSendParticlesExchange(par, direction);
                }

                template<class T_Particles>
                void handleIncoming(T_Particles& par, int32_t direction) const
                {
                    Environment<>::get().ParticleFactory().createTaskReceiveParticlesExchange(par, direction);
                }
            };

        } // namespace policies
    } // namespace particles
} // namespace pmacc
