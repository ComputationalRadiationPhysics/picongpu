/* Copyright 2014-2017 Rene Widera, Marco Garten, Alexander Grund,
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

#include "simulation_defines.hpp"
#include "traits/HasFlag.hpp"
#include "fields/Fields.def"
#include "math/MapTuple.hpp"

#include "Environment.hpp"
#include "communication/AsyncCommunication.hpp"
#include "particles/traits/GetIonizerList.hpp"
#include "particles/traits/FilterByFlag.hpp"
#include "particles/traits/GetPhotonCreator.hpp"
#include "particles/traits/ResolveAliasFromSpecies.hpp"
#include "particles/synchrotronPhotons/SynchrotronFunctions.hpp"
#include "particles/bremsstrahlung/Bremsstrahlung.hpp"
#include "particles/creation/creation.hpp"

#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>

#include <memory>


namespace picongpu
{

namespace particles
{

template<typename T_SpeciesType>
struct AssignNull
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    void operator()()
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        auto species = dc.get< SpeciesType >( FrameType::getName(), true );
        species = nullptr;
        dc.releaseData( FrameType::getName() );
    }
};

template<typename T_SpeciesType>
struct CreateSpecies
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    template<
        typename T_DeviceHeap,
        typename T_CellDescription
    >
    HINLINE void operator()(
        const std::shared_ptr<T_DeviceHeap>& deviceHeap,
        T_CellDescription* cellDesc
    ) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        dc.share(
            std::shared_ptr< ISimulationData >(
                new SpeciesType(
                    deviceHeap,
                    *cellDesc,
                    FrameType::getName()
                )
            )
        );
    }
};

template<typename T_SpeciesType>
struct CallCreateParticleBuffer
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    template<typename T_DeviceHeap>
    HINLINE void operator()(
        const std::shared_ptr<T_DeviceHeap>& deviceHeap
    ) const
    {
        log<picLog::MEMORY >("mallocMC: free slots for species %3%: %1% a %2%") %
            deviceHeap->getAvailableSlots(sizeof (FrameType)) %
            sizeof (FrameType) %
            FrameType::getName();

        DataConnector &dc = Environment<>::get().DataConnector();
        auto species = dc.get< SpeciesType >( FrameType::getName(), true );
        species->createParticleBuffer();
        dc.releaseData( FrameType::getName() );
    }
};

template<typename T_SpeciesType>
struct CallInit
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    HINLINE void operator()() const
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        auto species = dc.get< SpeciesType >( FrameType::getName(), true );
        species->init();
        dc.releaseData( FrameType::getName() );
    }
};

template<typename T_SpeciesType>
struct CallReset
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    HINLINE void operator()( const uint32_t currentStep )
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        auto species = dc.get< SpeciesType >( FrameType::getName(), true );
        species->reset( currentStep );
        dc.releaseData( FrameType::getName() );
    }
};

/** push a species
 *
 * push is only triggered for species with a pusher
 *
 * @tparam T_SpeciesType type of particle species that is checked
 */
template<typename T_SpeciesType>
struct PushSpecies
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    template<typename T_EventList>
    HINLINE void operator()(
        const uint32_t currentStep,
        const EventTask& eventInt,
        T_EventList& updateEvent
    ) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        auto species = dc.get< SpeciesType >( FrameType::getName(), true );

        __startTransaction(eventInt);
        species->update(currentStep);
        dc.releaseData( FrameType::getName() );
        EventTask ev = __endTransaction();
        updateEvent.push_back(ev);
    }
};

/** Communicate a species
 *
 * communication is only triggered for species with a pusher
 *
 * @tparam T_SpeciesType type of particle species that is checked
 */
template<typename T_SpeciesType>
struct CommunicateSpecies
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    template<typename T_EventList>
    HINLINE void operator()(
        T_EventList& updateEventList,
        T_EventList& commEventList
    ) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        auto species = dc.get< SpeciesType >( FrameType::getName(), true );

        EventTask updateEvent(*(updateEventList.begin()));

        updateEventList.pop_front();
        commEventList.push_back( communication::asyncCommunication(*species, updateEvent) );

        dc.releaseData( FrameType::getName() );
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
        EventTask& commEvent
    ) const
    {
        using EventList = std::list<EventTask>;
        EventList updateEventList;
        EventList commEventList;

        /* push all species */
        typedef typename PMacc::particles::traits::FilterByFlag
        <
            VectorAllSpecies,
            particlePusher<>
        >::type VectorSpeciesWithPusher;
        ForEach< VectorSpeciesWithPusher, particles::PushSpecies< bmpl::_1 > > pushSpecies;
        pushSpecies( currentStep, eventInt, forward(updateEventList) );

        /* join all push events */
        for (typename EventList::iterator iter = updateEventList.begin();
             iter != updateEventList.end();
             ++iter)
        {
            pushEvent += *iter;
        }

        /* call communication for all species */
        ForEach< VectorSpeciesWithPusher, particles::CommunicateSpecies< bmpl::_1> > communicateSpecies;
        communicateSpecies( forward(updateEventList), forward(commEventList) );

        /* join all communication events */
        for (typename EventList::iterator iter = commEventList.begin();
             iter != commEventList.end();
             ++iter)
        {
            commEvent += *iter;
        }
    }
};

/** Call an ionization method upon an ion species
 *
 * \tparam T_SpeciesType type of particle species that is going to be ionized with
 *                       ionization scheme T_SelectIonizer
 */
template< typename T_SpeciesType, typename T_SelectIonizer >
struct CallIonizationScheme
{
    using SpeciesType = T_SpeciesType;
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
     * \param cellDesc points to logical block information like dimension and cell sizes
     * \param currentStep The current time step
     */
    template<typename T_CellDescription>
    HINLINE void operator()(
        T_CellDescription* cellDesc,
        const uint32_t currentStep
    ) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        // alias for pointer on source species
        auto srcSpeciesPtr = dc.get< SpeciesType >( FrameType::getName(), true );
        // alias for pointer on destination species
        auto electronsPtr = dc.get< DestSpecies >( DestFrameType::getName(), true );

        // 3-dim vector : number of threads to be started in every dimension
        auto block = MappingDesc::SuperCellSize::toRT();

        AreaMapping< CORE + BORDER, MappingDesc > mapper( *cellDesc );
        /** kernelIonizeParticles
         *
         * calls the ionization model and handles that electrons are created correctly
         * while cycling through the particle frames
         *
         * kernel call : instead of name<<<blocks, threads>>> (args, ...)
         * "blocks" will be calculated from "this->cellDescription" and "CORE + BORDER"
         * "threads" is calculated from the previously defined vector "block"
         */
        PMACC_KERNEL( particles::ionization::KernelIonizeParticles{} )
            (mapper.getGridDim(), block)
            ( srcSpeciesPtr->getDeviceParticlesBox( ),
              electronsPtr->getDeviceParticlesBox( ),
              SelectIonizer(currentStep),
              mapper
            );
        /* fill the gaps in the created species' particle frames to ensure that only
         * the last frame is not completely filled but every other before is full
         */
        electronsPtr->fillAllGaps();

        dc.releaseData( FrameType::getName() );
        dc.releaseData( DestFrameType::getName() );
    }

};

/** Call all ionization schemes of an ion species
 *
 * Tests if species can be ionized and calls the kernels to do that
 *
 * \tparam T_SpeciesType type of particle species that is checked for ionization
 */
template< typename T_SpeciesType >
struct CallIonization
{
    using SpeciesType = T_SpeciesType;
    using FrameType = typename SpeciesType::FrameType;

    // SelectIonizer will be either the specified one or fallback: None
    using SelectIonizerList = typename traits::GetIonizerList< SpeciesType >::type;

    /** Functor implementation
     *
     * \tparam T_CellDescription contains the number of blocks and blocksize
     *                           that is later passed to the kernel
     * \param cellDesc points to logical block information like dimension and cell sizes
     * \param currentStep The current time step
     */
    template<typename T_CellDescription>
    HINLINE void operator()(
        T_CellDescription* cellDesc,
        const uint32_t currentStep
    ) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        // only if an ionizer has been specified, this is executed
        using hasIonizers = typename HasFlag< FrameType, ionizers<> >::type;
        if (hasIonizers::value)
        {
            ForEach< SelectIonizerList, CallIonizationScheme< SpeciesType, bmpl::_1 > > particleIonization;
            particleIonization( cellDesc, currentStep );
        }
    }

};

/** Handles the bremsstrahlung effect for electrons on ions.
 *
 * \tparam T_ElectronSpecies type of electron particle species
 */
template<typename T_ElectronSpecies>
struct CallBremsstrahlung
{
    using ElectronSpecies = T_ElectronSpecies;
    using ElectronFrameType = typename ElectronSpecies::FrameType;

    using IonSpecies = typename PMacc::particles::traits::ResolveAliasFromSpecies<
        ElectronSpecies,
        bremsstrahlungIons<>
    >::type;
    using PhotonSpecies = typename PMacc::particles::traits::ResolveAliasFromSpecies<
        ElectronSpecies,
        bremsstrahlungPhotons<>
    >::type;
    using PhotonFrameType = typename PhotonSpecies::FrameType;
    using BremsstrahlungFunctor = bremsstrahlung::Bremsstrahlung<
        IonSpecies,
        ElectronSpecies,
        PhotonSpecies
    >;

    /** Functor implementation
     *
     * \tparam T_CellDescription contains the number of blocks and blocksize
     *                           that is later passed to the kernel
     * \param cellDesc points to logical block information like dimension and cell sizes
     * \param currentStep the current time step
     */
    template<typename T_CellDescription, typename ScaledSpectrumMap>
    HINLINE void operator()(
        T_CellDescription* cellDesc,
        const uint32_t currentStep,
        const ScaledSpectrumMap& scaledSpectrumMap,
        const bremsstrahlung::GetPhotonAngle& photonAngle
    ) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        /* alias for pointer on source species */
        auto electronSpeciesPtr = dc.get< ElectronSpecies >( ElectronFrameType::getName(), true );
        /* alias for pointer on destination species */
        auto photonSpeciesPtr = dc.get< PhotonSpecies >( PhotonFrameType::getName(), true );

        const float_X targetZ = GetAtomicNumbers<IonSpecies>::type::numberOfProtons;

        using namespace bremsstrahlung;
        BremsstrahlungFunctor bremsstrahlungFunctor(
            scaledSpectrumMap.at(targetZ).getScaledSpectrumFunctor(),
            scaledSpectrumMap.at(targetZ).getStoppingPowerFunctor(),
            photonAngle.getPhotonAngleFunctor(),
            currentStep);

        creation::createParticlesFromSpecies(*electronSpeciesPtr, *photonSpeciesPtr, bremsstrahlungFunctor, cellDesc);

        dc.releaseData( ElectronFrameType::getName() );
        dc.releaseData( PhotonFrameType::getName() );
    }

};

/** Handles the synchrotron radiation emission of photons from electrons
 *
 * \tparam T_ElectronSpecies type of electron particle species
 */
template<typename T_ElectronSpecies>
struct CallSynchrotronPhotons
{
    using ElectronSpecies = T_ElectronSpecies;
    using ElectronFrameType = typename ElectronSpecies::FrameType;

    /* SelectedPhotonCreator will be either PhotonCreator or fallback: CreatorBase */
    using SelectedPhotonCreator = typename traits::GetPhotonCreator< ElectronSpecies >::type;
    using PhotonSpecies = typename SelectedPhotonCreator::PhotonSpecies;
    using PhotonFrameType = typename PhotonSpecies::FrameType;

    /** Functor implementation
     *
     * \tparam T_CellDescription contains the number of blocks and blocksize
     *                           that is later passed to the kernel
     * \param cellDesc points to logical block information like dimension and cell sizes
     * \param currentStep The current time step
     * \param synchrotronFunctions synchrotron functions wrapper object
     */
    template<typename T_CellDescription>
    HINLINE void operator()(
        T_CellDescription* cellDesc,
        const uint32_t currentStep,
        const synchrotronPhotons::SynchrotronFunctions& synchrotronFunctions
    ) const
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        /* alias for pointer on source species */
        auto electronSpeciesPtr = dc.get< ElectronSpecies >( ElectronFrameType::getName(), true );
        /* alias for pointer on destination species */
        auto photonSpeciesPtr = dc.get< PhotonSpecies >( PhotonFrameType::getName(), true );

        using namespace synchrotronPhotons;
        SelectedPhotonCreator photonCreator(
            synchrotronFunctions.getCursor(SynchrotronFunctions::first),
            synchrotronFunctions.getCursor(SynchrotronFunctions::second));

        creation::createParticlesFromSpecies(*electronSpeciesPtr, *photonSpeciesPtr, photonCreator, cellDesc);

        dc.releaseData( ElectronFrameType::getName() );
        dc.releaseData( PhotonFrameType::getName() );
    }

};

} // namespace particles
} // namespace picongpu
