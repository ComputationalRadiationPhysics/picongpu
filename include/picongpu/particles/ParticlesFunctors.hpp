/* Copyright 2014-2021 Rene Widera, Marco Garten, Alexander Grund,
 *                     Heiko Burau, Axel Huebl
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include <pmacc/traits/HasFlag.hpp>
#include "picongpu/fields/Fields.def"
#include <pmacc/math/MapTuple.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/communication/AsyncCommunication.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include "picongpu/particles/traits/GetIonizerList.hpp"
#if(PMACC_CUDA_ENABLED == 1)
#    include "picongpu/particles/bremsstrahlung/Bremsstrahlung.hpp"
#endif
#include "picongpu/particles/traits/GetPhotonCreator.hpp"
#include "picongpu/particles/synchrotronPhotons/SynchrotronFunctions.hpp"
#include "picongpu/particles/creation/creation.hpp"
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/particles/traits/ResolveAliasFromSpecies.hpp>
#include "picongpu/particles/flylite/IFlyLite.hpp"

#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>

#include <memory>


namespace picongpu
{
    namespace particles
    {
        /** assign nullptr to all attributes of a species
         *
         * @tparam T_SpeciesType type or name as boost::mpl::string of the species
         */
        template<typename T_SpeciesType>
        struct AssignNull
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            void operator()()
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName(), true);
                species = nullptr;
                dc.releaseData(FrameType::getName());
            }
        };

        /** create memory for the given species type
         *
         * @tparam T_SpeciesType type or name as boost::mpl::string of the species
         */
        template<typename T_SpeciesType>
        struct CreateSpecies
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            template<typename T_DeviceHeap, typename T_CellDescription>
            HINLINE void operator()(const std::shared_ptr<T_DeviceHeap>& deviceHeap, T_CellDescription* cellDesc) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                dc.consume(std::make_unique<SpeciesType>(deviceHeap, *cellDesc, FrameType::getName()));
            }
        };

        /** write memory statistics to the terminal
         *
         * @tparam T_SpeciesType type or name as boost::mpl::string of the species
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
                log<picLog::MEMORY>("mallocMC: free slots for species %3%: %1% a %2%")
                    % deviceHeap->getAvailableSlots(
                        cupla::manager::Device<cupla::AccDev>::get().current(),
                        cupla::manager::Stream<cupla::AccDev, cupla::AccStream>::get().stream(0),
                        sizeof(FrameType))
                    % sizeof(FrameType) % FrameType::getName();
#endif
            }
        };

        /** call method reset for the given species
         *
         * @tparam T_SpeciesType type or name as boost::mpl::string of the species to reset
         */
        template<typename T_SpeciesType>
        struct CallReset
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            HINLINE void operator()(const uint32_t currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName(), true);
                species->reset(currentStep);
                dc.releaseData(FrameType::getName());
            }
        };

        /** Allocate helper fields for FLYlite population kinetics for atomic physics
         *
         * energy histograms, rate matrix, etc.
         *
         * @tparam T_SpeciesType type or name as boost::mpl::string of ion species
         */
        template<typename T_SpeciesType>
        struct CallPopulationKineticsInit
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            using PopulationKineticsSolver =
                typename pmacc::traits::Resolve<typename GetFlagType<FrameType, populationKinetics<>>::type>::type;

            HINLINE void operator()(pmacc::DataSpace<simDim> gridSizeLocal) const
            {
                PopulationKineticsSolver flylite;
                flylite.init(gridSizeLocal, FrameType::getName());
            }
        };

        /** Calculate FLYlite population kinetics evolving one time step
         *
         * @tparam T_SpeciesType type or name as boost::mpl::string of ion species
         */
        template<typename T_SpeciesType>
        struct CallPopulationKinetics
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;

            using FrameType = typename SpeciesType::FrameType;

            using PopulationKineticsSolver =
                typename pmacc::traits::Resolve<typename GetFlagType<FrameType, populationKinetics<>>::type>::type;

            HINLINE void operator()(uint32_t currentStep) const
            {
                PopulationKineticsSolver flylite{};
                flylite.template update<SpeciesType>(FrameType::getName(), currentStep);
            }
        };

        /** push a species
         *
         * push is only triggered for species with a pusher
         *
         * @tparam T_SpeciesType type or name as boost::mpl::string of particle species that is checked
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
                auto species = dc.get<SpeciesType>(FrameType::getName(), true);

                __startTransaction(eventInt);
                species->update(currentStep);
                dc.releaseData(FrameType::getName());
                EventTask ev = __endTransaction();
                updateEvent.push_back(ev);
            }
        };

        /** Communicate a species
         *
         * communication is only triggered for species with a pusher
         *
         * @tparam T_SpeciesType type or name as boost::mpl::string of particle species that is checked
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
                auto species = dc.get<SpeciesType>(FrameType::getName(), true);

                EventTask updateEvent(*(updateEventList.begin()));

                updateEventList.pop_front();
                commEventList.push_back(communication::asyncCommunication(*species, updateEvent));

                dc.releaseData(FrameType::getName());
            }
        };

        /** update momentum, move and communicate all species */
        struct PushAllSpecies
        {
            /** push and communicate all species
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
                meta::ForEach<VectorSpeciesWithPusher, particles::PushSpecies<bmpl::_1>> pushSpecies;
                pushSpecies(currentStep, eventInt, updateEventList);

                /* join all push events */
                for(typename EventList::iterator iter = updateEventList.begin(); iter != updateEventList.end(); ++iter)
                {
                    pushEvent += *iter;
                }

                /* call communication for all species */
                meta::ForEach<VectorSpeciesWithPusher, particles::CommunicateSpecies<bmpl::_1>> communicateSpecies;
                communicateSpecies(updateEventList, commEventList);

                /* join all communication events */
                for(typename EventList::iterator iter = commEventList.begin(); iter != commEventList.end(); ++iter)
                {
                    commEvent += *iter;
                }
            }
        };

        /** Call an ionization method upon an ion species
         *
         * \tparam T_SpeciesType type or name as boost::mpl::string of particle species that is going to be ionized
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
             * \tparam T_CellDescription contains the number of blocks and blocksize
             *                           that is later passed to the kernel
             * \param cellDesc logical block information like dimension and cell sizes
             * \param currentStep The current time step
             */
            template<typename T_CellDescription>
            HINLINE void operator()(T_CellDescription cellDesc, const uint32_t currentStep) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                // alias for pointer on source species
                auto srcSpeciesPtr = dc.get<SpeciesType>(FrameType::getName(), true);
                // alias for pointer on destination species
                auto electronsPtr = dc.get<DestSpecies>(DestFrameType::getName(), true);

                SelectIonizer selectIonizer(currentStep);

                creation::createParticlesFromSpecies(*srcSpeciesPtr, *electronsPtr, selectIonizer, cellDesc);

                /* fill the gaps in the created species' particle frames to ensure that only
                 * the last frame is not completely filled but every other before is full
                 */
                electronsPtr->fillAllGaps();

                dc.releaseData(FrameType::getName());
                dc.releaseData(DestFrameType::getName());
            }
        };

        /** Call all ionization schemes of an ion species
         *
         * Tests if species can be ionized and calls the kernels to do that
         *
         * \tparam T_SpeciesType type or name as boost::mpl::string of particle species that is checked for ionization
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
             * \tparam T_CellDescription contains the number of blocks and blocksize
             *                           that is later passed to the kernel
             * \param cellDesc logical block information like dimension and cell sizes
             * \param currentStep The current time step
             */
            template<typename T_CellDescription>
            HINLINE void operator()(T_CellDescription cellDesc, const uint32_t currentStep) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                // only if an ionizer has been specified, this is executed
                using hasIonizers = typename HasFlag<FrameType, ionizers<>>::type;
                if(hasIonizers::value)
                {
                    meta::ForEach<SelectIonizerList, CallIonizationScheme<SpeciesType, bmpl::_1>> particleIonization;
                    particleIonization(cellDesc, currentStep);
                }
            }
        };

#if(PMACC_CUDA_ENABLED == 1)

        /** Handles the bremsstrahlung effect for electrons on ions.
         *
         * @tparam T_ElectronSpecies type or name as boost::mpl::string of electron particle species
         */
        template<typename T_ElectronSpecies>
        struct CallBremsstrahlung
        {
            using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;
            using ElectronFrameType = typename ElectronSpecies::FrameType;

            using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                typename pmacc::particles::traits::ResolveAliasFromSpecies<ElectronSpecies, bremsstrahlungIons<>>::
                    type>;
            using PhotonSpecies = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                typename pmacc::particles::traits::ResolveAliasFromSpecies<ElectronSpecies, bremsstrahlungPhotons<>>::
                    type>;
            using PhotonFrameType = typename PhotonSpecies::FrameType;
            using BremsstrahlungFunctor = bremsstrahlung::Bremsstrahlung<IonSpecies, ElectronSpecies, PhotonSpecies>;

            /** Functor implementation
             *
             * \tparam T_CellDescription contains the number of blocks and blocksize
             *                           that is later passed to the kernel
             * \param cellDesc logical block information like dimension and cell sizes
             * \param currentStep the current time step
             */
            template<typename T_CellDescription, typename ScaledSpectrumMap>
            HINLINE void operator()(
                T_CellDescription cellDesc,
                const uint32_t currentStep,
                const ScaledSpectrumMap& scaledSpectrumMap,
                const bremsstrahlung::GetPhotonAngle& photonAngle) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                /* alias for pointer on source species */
                auto electronSpeciesPtr = dc.get<ElectronSpecies>(ElectronFrameType::getName(), true);
                /* alias for pointer on destination species */
                auto photonSpeciesPtr = dc.get<PhotonSpecies>(PhotonFrameType::getName(), true);

                const float_X targetZ = GetAtomicNumbers<IonSpecies>::type::numberOfProtons;

                using namespace bremsstrahlung;
                BremsstrahlungFunctor bremsstrahlungFunctor(
                    scaledSpectrumMap.at(targetZ).getScaledSpectrumFunctor(),
                    scaledSpectrumMap.at(targetZ).getStoppingPowerFunctor(),
                    photonAngle.getPhotonAngleFunctor(),
                    currentStep);

                creation::createParticlesFromSpecies(
                    *electronSpeciesPtr,
                    *photonSpeciesPtr,
                    bremsstrahlungFunctor,
                    cellDesc);

                dc.releaseData(ElectronFrameType::getName());
                dc.releaseData(PhotonFrameType::getName());
            }
        };
#endif

        /** Handles the synchrotron radiation emission of photons from electrons
         *
         * @tparam T_ElectronSpecies type or name as boost::mpl::string of electron particle species
         */
        template<typename T_ElectronSpecies>
        struct CallSynchrotronPhotons
        {
            using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;
            using ElectronFrameType = typename ElectronSpecies::FrameType;

            /* SelectedPhotonCreator will be either PhotonCreator or fallback: CreatorBase */
            using SelectedPhotonCreator = typename traits::GetPhotonCreator<ElectronSpecies>::type;
            using PhotonSpecies = typename SelectedPhotonCreator::PhotonSpecies;
            using PhotonFrameType = typename PhotonSpecies::FrameType;

            /** Functor implementation
             *
             * \tparam T_CellDescription contains the number of blocks and blocksize
             *                           that is later passed to the kernel
             * \param cellDesc logical block information like dimension and cell sizes
             * \param currentStep The current time step
             * \param synchrotronFunctions synchrotron functions wrapper object
             */
            template<typename T_CellDescription>
            HINLINE void operator()(
                T_CellDescription cellDesc,
                const uint32_t currentStep,
                const synchrotronPhotons::SynchrotronFunctions& synchrotronFunctions) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                /* alias for pointer on source species */
                auto electronSpeciesPtr = dc.get<ElectronSpecies>(ElectronFrameType::getName(), true);
                /* alias for pointer on destination species */
                auto photonSpeciesPtr = dc.get<PhotonSpecies>(PhotonFrameType::getName(), true);

                using namespace synchrotronPhotons;
                SelectedPhotonCreator photonCreator(
                    synchrotronFunctions.getCursor(SynchrotronFunctions::first),
                    synchrotronFunctions.getCursor(SynchrotronFunctions::second));

                creation::createParticlesFromSpecies(*electronSpeciesPtr, *photonSpeciesPtr, photonCreator, cellDesc);

                dc.releaseData(ElectronFrameType::getName());
                dc.releaseData(PhotonFrameType::getName());
            }
        };

    } // namespace particles
} // namespace picongpu
