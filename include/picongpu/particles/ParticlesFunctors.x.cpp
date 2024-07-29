/* Copyright 2014-2024 Rene Widera, Marco Garten, Alexander Grund,
 *                     Heiko Burau, Axel Huebl, Sergei Bastrakov
 *                     Filip Optolowicz
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

#include "picongpu/particles/ParticlesFunctors.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/Fields.def"
#include "picongpu/particles/boundary/RemoveOuterParticles.hpp"
#include "picongpu/particles/creation/creation.hpp"
#include "picongpu/particles/synchrotron/AlgorithmSynchrotron.hpp"
#include "picongpu/particles/traits/GetIonizerList.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/communication/AsyncCommunication.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/particles/traits/ResolveAliasFromSpecies.hpp>
#include <pmacc/traits/HasFlag.hpp>

#include <memory>

namespace picongpu
{
    namespace particles
    {
        void PushAllSpecies::operator()(
            const uint32_t currentStep,
            const EventTask& eventInt,
            EventTask& pushEvent,
            EventTask& commEvent) const
        {
            using EventList = std::list<EventTask>;
            EventList updateEventList;
            EventList commEventList;

            /* push all species */
            using VectorSpeciesWithPusher =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, particlePusher<>>::type;
            meta::ForEach<VectorSpeciesWithPusher, PushSpecies<boost::mpl::_1>> pushSpecies;
            pushSpecies(currentStep, eventInt, updateEventList);

            /* join all push events */
            for(auto iter = updateEventList.begin(); iter != updateEventList.end(); ++iter)
            {
                pushEvent += *iter;
            }

            /* call communication for all species */
            meta::ForEach<VectorSpeciesWithPusher, particles::CommunicateSpecies<boost::mpl::_1>> communicateSpecies;
            communicateSpecies(updateEventList, commEventList);

            /* join all communication events */
            for(auto iter = commEventList.begin(); iter != commEventList.end(); ++iter)
            {
                commEvent += *iter;
            }
        }

        void RemoveOuterParticlesAllSpecies::operator()(const uint32_t currentStep) const
        {
            using VectorSpeciesWithPusher =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, particlePusher<>>::type;
            meta::ForEach<VectorSpeciesWithPusher, RemoveOuterParticles<boost::mpl::_1>> removeOuterParticles;
            removeOuterParticles(currentStep);
        }
    } // namespace particles
} // namespace picongpu
