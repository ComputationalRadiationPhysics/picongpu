/**
 * Copyright 2014-2016 Rene Widera, Marco Garten, Alexander Grund
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

#include "pmacc_types.hpp"
#include "simulation_defines.hpp"
#include <boost/mpl/if.hpp>
#include "traits/HasFlag.hpp"
#include "fields/Fields.def"
#include "math/MapTuple.hpp"
#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>

#include "communication/AsyncCommunication.hpp"
#include "particles/traits/GetIonizer.hpp"
#include "particles/traits/FilterByFlag.hpp"

namespace picongpu
{

namespace particles
{

template<typename T_SpeciesName>
struct AssignNull
{
    typedef T_SpeciesName SpeciesName;

    template<typename T_StorageTuple>
    void operator()(T_StorageTuple& tuple)
    {
        tuple[SpeciesName()] = NULL;
    }
};

template<typename T_SpeciesName>
struct CallDelete
{
    typedef T_SpeciesName SpeciesName;

    template<typename T_StorageTuple>
    void operator()(T_StorageTuple& tuple)
    {
        __delete(tuple[SpeciesName()]);
    }
};

template<typename T_SpeciesName>
struct CreateSpecies
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;

    template<typename T_StorageTuple, typename T_CellDescription>
    HINLINE void operator()(T_StorageTuple& tuple, T_CellDescription* cellDesc) const
    {
        tuple[SpeciesName()] = new SpeciesType(cellDesc->getGridLayout(), *cellDesc, SpeciesType::FrameType::getName());
    }
};

template<typename T_SpeciesName>
struct CallCreateParticleBuffer
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;

    template<typename T_StorageTuple>
    HINLINE void operator()(T_StorageTuple& tuple) const
    {

        typedef typename SpeciesType::FrameType FrameType;

        log<picLog::MEMORY >("mallocMC: free slots for species %3%: %1% a %2%") %
            mallocMC::getAvailableSlots(sizeof (FrameType)) %
            sizeof (FrameType) %
            FrameType::getName();

        tuple[SpeciesName()]->createParticleBuffer();
    }
};

template<typename T_SpeciesName>
struct CallInit
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;

    template<typename T_StorageTuple>
    HINLINE void operator()(T_StorageTuple& tuple,
                            FieldE* fieldE,
                            FieldB* fieldB,
                            FieldJ* fieldJ,
                            FieldTmp* fieldTmp) const
    {
        tuple[SpeciesName()]->init(*fieldE, *fieldB, *fieldJ, *fieldTmp);
    }
};

template<typename T_SpeciesName>
struct CallReset
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;

    template<typename T_StorageTuple>
    HINLINE void operator()(T_StorageTuple& tuple,
                            const uint32_t currentStep)
    {
        tuple[SpeciesName()]->reset(currentStep);
    }
};

/** push a species
 *
 * push is only triggered for species with a pusher
 *
 * @tparam T_SpeciesName name of particle species that is checked
 */
template<typename T_SpeciesName>
struct PushSpecies
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;
    typedef typename SpeciesType::FrameType FrameType;

    template<typename T_StorageTuple, typename T_EventList>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep,
                            const EventTask& eventInt,
                            T_EventList& updateEvent
                            ) const
    {
        PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);

        __startTransaction(eventInt);
        speciesPtr->update(currentStep);
        EventTask ev = __endTransaction();
        updateEvent.push_back(ev);
    }
};

/** Communicate a species
 *
 * communication is only triggered for species with a pusher
 *
 * @tparam T_SpeciesName name of particle species that is checked
 */
template<typename T_SpeciesName>
struct CommunicateSpecies
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;
    typedef typename SpeciesType::FrameType FrameType;

    template<typename T_StorageTuple, typename T_EventList>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            T_EventList& updateEventList,
                            T_EventList& commEventList
                            ) const
    {
        EventTask updateEvent(*(updateEventList.begin()));

        updateEventList.pop_front();
        commEventList.push_back( communication::asyncCommunication(*tuple[SpeciesName()], updateEvent) );
    }
};

/** update momentum, move and communicate all species */
struct PushAllSpecies
{

    /** push and communicate all species
     *
     * @tparam T_SpeciesStorage type of the speciesStorage
     * @param speciesStorage struct with all species (e.g. `PMacc::math::MapTuple`)
     * @param currentStep current simulation step
     * @param pushEvent[out] grouped event that marks the end of the species push
     * @param commEvent[out] grouped event that marks the end of the species communication
     */
    template<typename T_SpeciesStorage>
    HINLINE void operator()(
                            T_SpeciesStorage& speciesStorage,
                            const uint32_t currentStep,
                            const EventTask& eventInt,
                            EventTask& pushEvent,
                            EventTask& commEvent
                            ) const
    {
        typedef std::list<EventTask> EventList;
        EventList updateEventList;
        EventList commEventList;

        /* push all species */
        typedef typename PMacc::particles::traits::FilterByFlag
        <
            VectorAllSpecies,
            particlePusher<>
        >::type VectorSpeciesWithPusher;
        ForEach<VectorSpeciesWithPusher, particles::PushSpecies<bmpl::_1>, MakeIdentifier<bmpl::_1> > pushSpecies;
        pushSpecies(forward(speciesStorage), currentStep, eventInt, forward(updateEventList));

        /* join all push events */
        for (typename EventList::iterator iter = updateEventList.begin();
             iter != updateEventList.end();
             ++iter)
        {
            pushEvent += *iter;
        }

        /* call communication for all species */
        ForEach<VectorSpeciesWithPusher, particles::CommunicateSpecies<bmpl::_1>, MakeIdentifier<bmpl::_1> > communicateSpecies;
        communicateSpecies(forward(speciesStorage), forward(updateEventList), forward(commEventList));

        /* join all communication events */
        for (typename EventList::iterator iter = commEventList.begin();
             iter != commEventList.end();
             ++iter)
        {
            commEvent += *iter;
        }
    }
};

/** \struct CallIonization
 *
 * \brief Tests if species can be ionized and calls the kernel to do that
 *
 * \tparam T_SpeciesName name of particle species that is checked for ionization
 */
template<typename T_SpeciesName>
struct CallIonization
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;
    typedef typename SpeciesType::FrameType FrameType;
    /* SelectIonizer will be either the specified one or fallback: None */
    typedef typename picongpu::traits::GetIonizer<SpeciesType>::type SelectIonizer;

    /** Functor implementation
     *
     * \tparam T_StorageStuple contains info about the particle species
     * \tparam T_CellDescription contains the number of blocks and blocksize
     *                           that is later passed to the kernel
     * \param tuple An n-tuple containing the type-info of multiple particle species
     * \param cellDesc points to logical block information like dimension and cell sizes
     * \param currentStep The current time step
     */
    template<typename T_StorageTuple, typename T_CellDescription>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            T_CellDescription* cellDesc,
                            const uint32_t currentStep
                            ) const
    {
        /* only if an ionizer has been specified, this is executed */
        typedef typename HasFlag<FrameType, ionizer<> >::type hasIonizer;
        if (hasIonizer::value)
        {

            /* define the type of the species to be created
             * from inside the ionization model specialization
             */
            typedef typename SelectIonizer::DestSpecies DestSpecies;
            /* alias for pointer on source species */
            PMACC_AUTO(srcSpeciesPtr, tuple[SpeciesName()]);
            /* alias for pointer on destination species */
            PMACC_AUTO(electronsPtr,  tuple[typename MakeIdentifier<DestSpecies>::type()]);

            /* 3-dim vector : number of threads to be started in every dimension */
            dim3 block( MappingDesc::SuperCellSize::toRT().toDim3() );

            /** kernelIonizeParticles
             * \brief calls the ionization model and handles that electrons are created correctly
             *        while cycling through the particle frames
             *
             * kernel call : instead of name<<<blocks, threads>>> (args, ...)
             * "blocks" will be calculated from "this->cellDescription" and "CORE + BORDER"
             * "threads" is calculated from the previously defined vector "block"
             */
            __picKernelArea( particles::ionization::kernelIonizeParticles, *cellDesc, CORE + BORDER )
                (block)
                ( srcSpeciesPtr->getDeviceParticlesBox( ),
                  electronsPtr->getDeviceParticlesBox( ),
                  SelectIonizer(currentStep)
                );
            /* fill the gaps in the created species' particle frames to ensure that only
             * the last frame is not completely filled but every other before is full
             */
            electronsPtr->fillAllGaps();

        }
    }

}; // struct CallIonization

} //namespace particles

} //namespace picongpu
