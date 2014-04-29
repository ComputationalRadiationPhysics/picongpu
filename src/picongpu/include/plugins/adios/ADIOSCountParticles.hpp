/**
 * Copyright 2014 Felix Schmitt
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

#include <mpi.h>

#include "types.h"
#include "simulation_types.hpp"
#include "plugins/adios/ADIOSWriter.def"

#include "plugins/ISimulationPlugin.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include "compileTime/conversion/MakeSeq.hpp"

#include "RefWrapper.hpp"
#include <boost/type_traits.hpp>

#include "plugins/output/WriteSpeciesCommon.hpp"
#include "plugins/kernel/CopySpecies.kernel"
#include "mappings/kernel/AreaMapping.hpp"

#include "traits/PICToAdios.hpp"
#include "plugins/adios/writer/ParticleAttributeSize.hpp"
#include "compileTime/conversion/RemoveFromSeq.hpp"

namespace picongpu
{

namespace adios
{
using namespace PMacc;



/** Count number of particles for a species
 *
 * @tparam T_Species type of species
 *
 */
template< typename T_Species >
struct ADIOSCountParticles
{
public:

    typedef T_Species ThisSpecies;
    typedef typename ThisSpecies::FrameType FrameType;
    typedef typename FrameType::ParticleDescription ParticleDescription;
    typedef typename FrameType::ValueTypeSeq ParticleAttributeList;

    /* delete multiMask and localCellIdx in adios particle*/
    typedef bmpl::vector<multiMask,localCellIdx> TypesToDelete;
    typedef typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type ParticleCleanedAttributeList;

    /* add globalCellIdx for adios particle*/
    typedef typename MakeSeq<
            ParticleCleanedAttributeList,
            globalCellIdx<globalCellIdx_pic>
    >::type ParticleNewAttributeList;

    typedef
    typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type
    NewParticleDescription;

    typedef Frame<OperatorCreateVectorBox, NewParticleDescription> AdiosFrameType;

    HINLINE void operator()(RefWrapper<ThreadParams*> params,
                            const std::string subGroup,
                            const DomainInformation domInfo)
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        GridController<simDim>& gc = Environment<simDim>::get().GridController();
        uint64_t mpiSize = gc.getGlobalSize();
        uint64_t mpiRank = gc.getGlobalRank();

        /* load particle without copy particle data to host */
        ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::getName(), true));

        /* count total number of particles on the device */
        uint64_cu totalNumParticles = 0;
        totalNumParticles = PMacc::CountParticles::countOnDevice < CORE + BORDER > (
                                                                                    *speciesTmp,
                                                                                    *(params.get()->cellDescription),
                                                                                    domInfo.localDomainOffset,
                                                                                    domInfo.domainSize);

        /* MPI_Allgather to compute global size and my offset */
        uint64_t myNumParticles = totalNumParticles;
        uint64_t allNumParticles[mpiSize];
        uint64_t globalNumParticles = 0;
        uint64_t myParticleOffset = 0;

        MPI_CHECK(MPI_Allgather(
                &myNumParticles, 1, MPI_UNSIGNED_LONG_LONG,
                allNumParticles, 1, MPI_UNSIGNED_LONG_LONG,
                gc.getCommunicator().getMPIComm()));

        for (uint64_t i = 0; i < mpiSize; ++i)
        {
            globalNumParticles += allNumParticles[i];
            if (i < mpiRank)
                myParticleOffset += allNumParticles[i];
        }

        if (myNumParticles > 0)
        {
            /* iterate over all attributes of this species */
            ForEach<typename AdiosFrameType::ValueTypeSeq, adios::ParticleAttributeSize<bmpl::_1> > attributeSize;
            attributeSize(params, (FrameType::getName() + std::string("/") + subGroup).c_str(),
                    myNumParticles, globalNumParticles, myParticleOffset);
        }

        /* define adios var for species index/info table */
        {
            const size_t localTableSize = 5;
            traits::PICToAdios<uint64_t> adiosIndexType;

            std::stringstream indexVarSizeStr;
            indexVarSizeStr << localTableSize;

            std::stringstream indexVarGlobalSizeStr;
            indexVarGlobalSizeStr << localTableSize * gc.getGlobalSize();

            std::stringstream indexVarOffsetStr;
            indexVarOffsetStr << localTableSize * gc.getGlobalRank();

            int64_t adiosSpeciesIndexVar = adios_define_var(
                params.get()->adiosGroupHandle,
                (params.get()->adiosBasePath + std::string(ADIOS_PATH_PARTICLES) +
                    FrameType::getName() + std::string("/") + subGroup +
                    std::string("particles_info")).c_str(),
                NULL,
                adiosIndexType.type,
                indexVarSizeStr.str().c_str(),
                indexVarGlobalSizeStr.str().c_str(),
                indexVarOffsetStr.str().c_str());

            params.get()->adiosSpeciesIndexVarIds.push_back(adiosSpeciesIndexVar);

            params.get()->adiosGroupSize += sizeof(uint64_t) * localTableSize * gc.getGlobalSize();
        }
    }
};


} //namspace adios

} //namespace picongpu
