/**
 * Copyright 2014 Rene Widera, Marco Garten
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

#include "types.h"
#include "simulation_defines.hpp"
#include <boost/mpl/if.hpp>
#include "traits/HasFlag.hpp"
#include "fields/Fields.def"
#include "math/MapTuple.hpp"
#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>

#include "particles/traits/GetIonizer.hpp"

namespace picongpu
{

namespace particles
{

template<typename T_Type>
struct AssignNull
{
    typedef T_Type SpeciesName;

    template<typename T_StorageTuple>
    void operator()(T_StorageTuple& tuple)
    {
        tuple[SpeciesName()] = NULL;
    }
};

template<typename T_Type>
struct CallDelete
{
    typedef T_Type SpeciesName;

    template<typename T_StorageTuple>
    void operator()(T_StorageTuple& tuple)
    {
        __delete(tuple[SpeciesName()]);
    }
};

template<typename T_Type>
struct CreateSpecies
{
    typedef T_Type SpeciesName;
    typedef typename T_Type::type SpeciesType;

    template<typename T_StorageTuple, typename T_CellDescription>
    HINLINE void operator()(T_StorageTuple& tuple, T_CellDescription* cellDesc) const
    {
        tuple[SpeciesName()] = new SpeciesType(cellDesc->getGridLayout(), *cellDesc, SpeciesType::FrameType::getName());
    }
};

template<typename T_Type>
struct CallCreateParticleBuffer
{
    typedef T_Type SpeciesName;
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

template<typename T_Type>
struct CallInit
{
    typedef T_Type SpeciesName;
    typedef typename T_Type::type SpeciesType;

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

template<typename T_SpeciesName>
struct CallUpdate
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;
    typedef typename SpeciesType::FrameType FrameType;

    template<typename T_StorageTuple, typename T_Event>
    HINLINE void operator()(
                            T_StorageTuple& tuple,
                            const uint32_t currentStep,
                            const T_Event eventInt,
                            T_Event& updateEvent,
                            T_Event& commEvent
                            ) const
    {
        typedef typename HasFlag<FrameType, particlePusher<> >::type hasPusher;
        if (hasPusher::value)
        {
            PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);

            __startTransaction(eventInt);
            speciesPtr->update(currentStep);
            commEvent += speciesPtr->asyncCommunication(__getTransactionEvent());
            updateEvent += __endTransaction();
        }
    }
};

/* Tests if species can be ionized and calls the function to do that */
template<typename T_SpeciesName>
struct CallIonization
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;
    
    typedef typename GetIonizer<SpeciesType>::type SelectIonizer;

    /* describes the instance of CallIonization */
    template<typename T_StorageTuple>
    HINLINE void operator()(
                        T_StorageTuple& tuple,
                        const uint32_t currentStep
                        ) const
    {
        
        /* alias for pointer on source species */
        PMACC_AUTO(speciesPtr, tuple[SpeciesName()]);
        /* instance of particle ionizer that was flagged in speciesDefinition.param */
        SelectIonizer myIonizer;
        myIonizer(*speciesPtr, tuple, currentStep);
        
    }

}; // struct CallIonization

} //namespace particles

} //namespace picongpu
