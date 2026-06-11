/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace particles
    {
        namespace policies
        {
            /**
             * Policy for @see HandleGuardRegion that moves particles from guard cells to exchange buffers
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
