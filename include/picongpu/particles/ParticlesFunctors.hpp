/* Copyright 2014-2023 Rene Widera, Marco Garten, Alexander Grund,
 *                     Heiko Burau, Axel Huebl, Sergei Bastrakov
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

#include "picongpu/fields/Fields.def"
#include "picongpu/particles/boundary/RemoveOuterParticles.hpp"
#include "picongpu/particles/creation/creation.hpp"
#include "picongpu/particles/traits/GetIonizerList.hpp"

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
                auto alpakaStream = pmacc::eventSystem::getEventStream(ITask::TASK_DEVICE)->getCudaStream();
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

        /** Remove all particles of the species that are outside the respective boundaries
         *
         * Must be called only for species with a pusher
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked
         */
        template<typename T_SpeciesType>
        struct RemoveOuterParticles
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            HINLINE void operator()(const uint32_t currentStep) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                boundary::removeOuterParticles(*species, currentStep);
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

        //! Remove all particles of all species with pusher flag that are outside the respective boundaries
        struct RemoveOuterParticlesAllSpecies
        {
            /** Remove all external particles
             *
             * @param currentStep current simulation step
             */
            HINLINE void operator()(const uint32_t currentStep) const
            {
                using VectorSpeciesWithPusher =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, particlePusher<>>::type;
                meta::ForEach<VectorSpeciesWithPusher, RemoveOuterParticles<boost::mpl::_1>> removeOuterParticles;
                removeOuterParticles(currentStep);
            }
        };

        /** Call an ionization method upon an ion species
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is going to be ionized
         * with ionization scheme T_SelectIonizer
         */
        template<typename T_SpeciesType, typename T_SelectIonizer>
        struct CallIonizationScheme
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using SelectIonizer = T_SelectIonizer;
            using FrameType = typename SpeciesType::FrameType;

            /* define the type of the species to be created
             * from inside the ionization model specialization
             */
            using DestSpecies = typename SelectIonizer::DestSpecies;
            using DestFrameType = typename DestSpecies::FrameType;

            /** Functor implementation
             *
             * @tparam T_CellDescription contains the number of blocks and blocksize
             *                           that is later passed to the kernel
             * @param cellDesc logical block information like dimension and cell sizes
             * @param currentStep The current time step
             */
            template<typename T_CellDescription>
            HINLINE void operator()(T_CellDescription cellDesc, const uint32_t currentStep) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                // alias for pointer on source species
                auto srcSpeciesPtr = dc.get<SpeciesType>(FrameType::getName());
                // alias for pointer on destination species
                auto electronsPtr = dc.get<DestSpecies>(DestFrameType::getName());

                SelectIonizer selectIonizer(currentStep);

                creation::createParticlesFromSpecies(*srcSpeciesPtr, *electronsPtr, selectIonizer, cellDesc);

                /* fill the gaps in the created species' particle frames to ensure that only
                 * the last frame is not completely filled but every other before is full
                 */
                electronsPtr->fillAllGaps();
            }
        };

        /** Call all ionization schemes of an ion species
         *
         * Tests if species can be ionized and calls the kernels to do that
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked for ionization
         */
        template<typename T_SpeciesType>
        struct CallIonization
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            // SelectIonizer will be either the specified one or fallback: None
            using SelectIonizerList = typename traits::GetIonizerList<SpeciesType>::type;

            /** Functor implementation
             *
             * @tparam T_CellDescription contains the number of blocks and blocksize
             *                           that is later passed to the kernel
             * @param cellDesc logical block information like dimension and cell sizes
             * @param currentStep The current time step
             */
            template<typename T_CellDescription>
            HINLINE void operator()(T_CellDescription cellDesc, const uint32_t currentStep) const
            {
                // only if an ionizer has been specified, this is executed
                using hasIonizers = typename HasFlag<FrameType, ionizers<>>::type;
                if(hasIonizers::value)
                {
                    meta::ForEach<SelectIonizerList, CallIonizationScheme<SpeciesType, boost::mpl::_1>>
                        particleIonization;
                    particleIonization(cellDesc, currentStep);
                }
            }
        };

    } // namespace particles
} // namespace picongpu
