/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/communication/AsyncCommunication.hpp"
#include "pmacc/particles/ParticlesBase.hpp"
#include "pmacc/types.hpp"

#include <type_traits>

namespace pmacc
{
    /**
     * Trait that should return true if T is a particle species
     */
    template<typename T>
    struct IsParticleSpecies
    {
        static inline constexpr bool value = std::is_same_v<typename T::SimulationDataTag, ParticlesTag>;
    };

    namespace communication
    {
        template<typename T_Data>
        struct AsyncCommunicationImpl<T_Data, Bool2Type<IsParticleSpecies<T_Data>::value>>
        {
            template<class T_Particles>
            EventTask operator()(T_Particles& par, EventTask event) const
            {
                EventTask ret;
                eventSystem::startTransaction(event);
                Environment<>::get().ParticleFactory().createTaskParticlesReceive(par);
                ret = eventSystem::endTransaction();

                eventSystem::startTransaction(event);
                Environment<>::get().ParticleFactory().createTaskParticlesSend(par);
                ret += eventSystem::endTransaction();
                return ret;
            }
        };

    } // namespace communication
} // namespace pmacc
