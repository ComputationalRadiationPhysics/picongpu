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

#include "picongpu/defines.hpp"
#include "picongpu/particles/boundary/Apply.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/communication/AsyncCommunication.hpp>
#include <pmacc/eventSystem/Manager.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <cstdint>

namespace picongpu
{
    namespace particles
    {
        /** Push a species and apply boundary conditions
         *
         * Both operations only affect species with a pusher
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked
         */
        template<typename T_SpeciesType>
        struct PushSpecies
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            template<typename T_EventList>
            HINLINE void operator()(const uint32_t currentStep, const EventTask& eventInt, T_EventList& updateEvent)
                const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());

                eventSystem::startTransaction(eventInt);
                species->update(currentStep);
                // No need to wait here
                species->applyBoundary(currentStep);
                EventTask ev = eventSystem::endTransaction();
                updateEvent.push_back(ev);
            }
        };

        /** Communicate a species
         *
         * communication is only triggered for species with a pusher
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked
         */
        template<typename T_SpeciesType>
        struct CommunicateSpecies
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            template<typename T_EventList>
            HINLINE void operator()(T_EventList& updateEventList, T_EventList& commEventList) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());

                EventTask updateEvent(*(updateEventList.begin()));

                updateEventList.pop_front();
                commEventList.push_back(communication::asyncCommunication(*species, updateEvent));
            }
        };

        //! Push, apply boundaries, and communicate all species with pusher flag
        struct PushAllSpecies
        {
            /** Process and communicate all species
             *
             * @param currentStep current simulation step
             * @param pushEvent[out] grouped event that marks the end of the species push
             * @param commEvent[out] grouped event that marks the end of the species communication
             */
            HINLINE void operator()(
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
                meta::ForEach<VectorSpeciesWithPusher, particles::CommunicateSpecies<boost::mpl::_1>>
                    communicateSpecies;
                communicateSpecies(updateEventList, commEventList);

                /* join all communication events */
                for(auto iter = commEventList.begin(); iter != commEventList.end(); ++iter)
                {
                    commEvent += *iter;
                }
            }
        };
    } // namespace particles

    namespace simulation
    {
        namespace stage
        {
            void ParticlePush::operator()(uint32_t const step, pmacc::EventTask& commEvent) const
            {
                pmacc::EventTask initEvent = eventSystem::getTransactionEvent();
                pmacc::EventTask updateEvent;
                picongpu::particles::PushAllSpecies pushAllSpecies;
                pushAllSpecies(step, initEvent, updateEvent, commEvent);
                eventSystem::setTransactionEvent(updateEvent);
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
