/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <cstdint>

namespace pmacc
{
    namespace particles
    {
        namespace policies
        {
            /**
             * Policy for HandleGuardParticles that removes all particles from guard cells
             */
            struct DeleteParticles
            {
                template<class T_Particles>
                void handleOutgoing(T_Particles& par, int32_t direction) const
                {
                    par.deleteGuardParticles(direction);
                }

                template<class T_Particles>
                void handleIncoming(T_Particles& par, int32_t direction) const
                {
                }
            };

        } // namespace policies
    } // namespace particles
} // namespace pmacc
