/* Copyright 2013-2023 Heiko Burau, Rene Widera, Alexander Grund
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
        inline static constexpr bool value = std::is_same_v<typename T::SimulationDataTag, ParticlesTag>;
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
