/**
 * Copyright 2014 Rene Widera
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

namespace picongpu
{

namespace particles
{

template<typename T_Type>
struct AssignNull
{
    typedef T_Type SpeciesName;

    template<typename T_StorageTupel>
    void operator()(T_StorageTupel& tupel)
    {
        tupel[SpeciesName()] = NULL;
    }
};

template<typename T_Type>
struct CallDelete
{
    typedef T_Type SpeciesName;

    template<typename T_StorageTupel>
    void operator()(T_StorageTupel& tupel)
    {
        __delete(tupel[SpeciesName()]);
    }
};

template<typename T_Type>
struct CreateSpecies
{
    typedef T_Type SpeciesName;
    typedef typename T_Type::type SpeciesType;

    template<typename T_StorageTupel, typename T_CellDescription>
    HINLINE void operator()(T_StorageTupel& tupel, T_CellDescription* cellDesc) const
    {
        tupel[SpeciesName()] = new SpeciesType(cellDesc->getGridLayout(), *cellDesc, SpeciesType::FrameType::getName());
    }
};

template<typename T_Species>
struct GetMemoryFactor
{
    typedef bmpl::integral_c<size_t, T_Species::FrameType::memFactor> type;
};

template<typename T_Type>
struct CallCreateParticleBuffer
{
    typedef T_Type SpeciesName;
    typedef typename SpeciesName::type SpeciesType;

    template<typename T_StorageTupel>
    HINLINE void operator()(T_StorageTupel& tupel, const size_t freeGpuMem) const
    {

        const size_t myMemFactor = SpeciesType::FrameType::memFactor;
        typedef typename bmpl::accumulate<
            VectorAllSpecies,
            bmpl::integral_c<size_t, 0>,
            bmpl::plus<bmpl::_1, GetMemoryFactor<bmpl::_2> >,
            >::type AccumulatedMemFactors;

        const size_t accumulatedMemFactors = AccumulatedMemFactors::value;
        size_t byte = freeGpuMem * myMemFactor / accumulatedMemFactors;

        log<picLog::MEMORY > ("create %1% MiB for species %2%") %
            (byte / 1024 / 1024) %
            SpeciesType::FrameType::getName();

        tupel[SpeciesName()]->createParticleBuffer(byte);
    }
};

template<typename T_Type>
struct CallInit
{
    typedef T_Type SpeciesName;
    typedef typename T_Type::type SpeciesType;

    template<typename T_StorageTupel>
    HINLINE void operator()(T_StorageTupel& tupel,
                            FieldE* fieldE,
                            FieldB* fieldB,
                            FieldJ* fieldJ,
                            FieldTmp* fieldTmp) const
    {
        tupel[SpeciesName()]->init(*fieldE, *fieldB, *fieldJ, *fieldTmp);
    }
};

template<typename T_SpeciesName>
struct CallReset
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;

    template<typename T_StorageTupel>
    HINLINE void operator()(T_StorageTupel& tupel,
                            const uint32_t currentStep)
    {
        tupel[SpeciesName()]->reset(currentStep);
    }
};

template<typename T_SpeciesName>
struct CallUpdate
{
    typedef T_SpeciesName SpeciesName;
    typedef typename SpeciesName::type SpeciesType;
    typedef typename SpeciesType::FrameType FrameType;

    template<typename T_StorageTupel, typename T_Event>
    HINLINE void operator()(
                            T_StorageTupel& tupel,
                            const uint32_t currentStep,
                            const T_Event eventInt,
                            T_Event& updateEvent,
                            T_Event& commEvent
                            ) const
    {
        typedef typename HasFlag<FrameType, particlePusher<> >::type hasPusher;
        if (hasPusher::value)
        {
            PMACC_AUTO(speciePtr, tupel[SpeciesName()]);

            __startTransaction(eventInt);
            speciePtr->update(currentStep);
            commEvent += speciePtr->asyncCommunication(__getTransactionEvent());
            updateEvent += __endTransaction();
        }
    }
};

} //namespace particles

} //namespace picongpu
