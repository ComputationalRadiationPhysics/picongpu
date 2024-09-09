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

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/communication/AsyncCommunication.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/particles/traits/ResolveAliasFromSpecies.hpp>
#include <pmacc/traits/HasFlag.hpp>

#include <memory>

namespace picongpu
{
    namespace particles
    {
        /** assign nullptr to all attributes of a species
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the species
         */
        template<typename T_SpeciesType>
        struct AssignNull
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            void operator()()
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                species = nullptr;
            }
        };

        /** create memory for the given species type
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the species
         */
        template<typename T_SpeciesType>
        struct CreateSpecies
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            template<typename T_DeviceHeap, typename T_CellDescription>
            HINLINE void operator()(std::shared_ptr<T_DeviceHeap> const& deviceHeap, T_CellDescription* cellDesc) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                dc.consume(std::make_unique<SpeciesType>(deviceHeap, *cellDesc, FrameType::getName()));
            }
        };

        /** write memory statistics to the terminal
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the species
         */
        template<typename T_SpeciesType>
        struct LogMemoryStatisticsForSpecies
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            template<typename T_DeviceHeap>
            HINLINE void operator()(const std::shared_ptr<T_DeviceHeap>& deviceHeap) const
            {
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
                auto alpakaStream = pmacc::eventSystem::getComputeDeviceQueue(ITask::TASK_DEVICE)->getAlpakaQueue();
                log<picLog::MEMORY>("mallocMC: free slots for species %3%: %1% a %2%")
                    % deviceHeap->getAvailableSlots(
                        manager::Device<ComputeDevice>::get().current(),
                        alpakaStream,
                        sizeof(FrameType))
                    % sizeof(FrameType) % FrameType::getName();
#endif
            }
        };

        /** call method reset for the given species
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the species to reset
         */
        template<typename T_SpeciesType>
        struct CallReset
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            HINLINE void operator()(const uint32_t currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                species->reset(currentStep);
            }
        };

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
} // namespace picongpu
