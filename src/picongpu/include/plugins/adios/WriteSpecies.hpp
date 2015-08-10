/**
 * Copyright 2014-2015 Rene Widera, Felix Schmitt, Axel Huebl,
 *                     Alexander Grund
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

#include <boost/type_traits.hpp>

#include "plugins/output/WriteSpeciesCommon.hpp"
#include "plugins/kernel/CopySpecies.kernel"
#include "mappings/kernel/AreaMapping.hpp"

#include "plugins/adios/writer/ParticleAttribute.hpp"
#include "compileTime/conversion/RemoveFromSeq.hpp"
#include "particles/ParticleDescription.hpp"

#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"
#include "particles/memory/buffers/MallocMCBuffer.hpp"

namespace picongpu
{

namespace adios
{
using namespace PMacc;

/** Write copy particle to host memory and dump to ADIOS file
 *
 * @tparam T_Species type of species
 *
 */
template< typename T_Species >
struct WriteSpecies
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

    template<typename Space>
    HINLINE void operator()(ThreadParams* params,
                            const Space particleOffset)
    {
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) write species: %1%") % AdiosFrameType::getName();
        DataConnector &dc = Environment<>::get().DataConnector();
        /* load particle without copy particle data to host */
        ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::getName(), true));

        /* count total number of particles on the device */
        log<picLog::INPUT_OUTPUT > ("ADIOS:   (begin) count particles: %1%") % AdiosFrameType::getName();
        uint64_cu totalNumParticles = 0;
        totalNumParticles = PMacc::CountParticles::countOnDevice < CORE + BORDER > (
                                                                                    *speciesTmp,
                                                                                    *(params->cellDescription),
                                                                                    params->localWindowToDomainOffset,
                                                                                    params->window.localDimensions.size);
        log<picLog::INPUT_OUTPUT > ("ADIOS:   ( end ) count particles: %1% = %2%") % AdiosFrameType::getName() % totalNumParticles;

        AdiosFrameType hostFrame;

        /* malloc host memory */
        log<picLog::INPUT_OUTPUT > ("ADIOS:   (begin) malloc host memory: %1%") % AdiosFrameType::getName();
        ForEach<typename AdiosFrameType::ValueTypeSeq, MallocHostMemory<bmpl::_1> > mallocMem;
        mallocMem(forward(hostFrame), totalNumParticles);
        log<picLog::INPUT_OUTPUT > ("ADIOS:   ( end ) malloc host memory: %1%") % AdiosFrameType::getName();

        if (totalNumParticles > 0)
        {
            log<picLog::INPUT_OUTPUT > ("ADIOS:   (begin) copy particle host (with hierarchy) to host (without hierarchy): %1%") % AdiosFrameType::getName();
            typedef bmpl::vector< typename GetPositionFilter<simDim>::type > usedFilters;
            typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
            MyParticleFilter filter;
            /* activate filter pipeline if moving window is activated */
            filter.setStatus(MovingWindow::getInstance().isSlidingWindowActive());
            filter.setWindowPosition(params->localWindowToDomainOffset,
                                     params->window.localDimensions.size);

            DataConnector &dc = Environment<>::get().DataConnector();
            MallocMCBuffer& mallocMCBuffer = dc.getData<MallocMCBuffer> (MallocMCBuffer::getName(),true);

            int globalParticleOffset = 0;
            AreaMapping < CORE + BORDER, MappingDesc > mapper(*(params->cellDescription));

            ConcatListOfFrames concatListOfFrames(mapper.getGridDim());

            concatListOfFrames(
                                globalParticleOffset,
                                hostFrame,
                                speciesTmp->getHostParticlesBox(mallocMCBuffer.getOffset()),
                                filter,
                                particleOffset, /*relative to data domain (not to physical domain)*/
                                mapper
                                );
            dc.releaseData(MallocMCBuffer::getName());
            /* this costs a little bit of time but adios writing is slower */
            assert((uint64_cu) globalParticleOffset == totalNumParticles);
        }
        /* dump to adios file */
        ForEach<typename AdiosFrameType::ValueTypeSeq, adios::ParticleAttribute<bmpl::_1> > writeToAdios;
        writeToAdios(params, forward(hostFrame), totalNumParticles);

        /* free host memory */
        ForEach<typename AdiosFrameType::ValueTypeSeq, FreeHostMemory<bmpl::_1> > freeMem;
        freeMem(forward(hostFrame));
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) writing species: %1%") % AdiosFrameType::getName();

        /* write species counter table to adios file */
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) writing particle index table for %1%") % AdiosFrameType::getName();
        {
            GridController<simDim>& gc = Environment<simDim>::get().GridController();

            const size_t pos_offset = 2;

            /* particlesMetaInfo = (num particles, scalar position, particle offset x, y, z) */
            uint64_t particlesMetaInfo[5] = {totalNumParticles, gc.getScalarPosition(), 0, 0, 0};
            for (size_t d = 0; d < simDim; ++d)
                particlesMetaInfo[pos_offset + d] = particleOffset[d];

            /* prevent that top (y) gpus have negative value here */
            if (gc.getPosition().y() == 0)
                particlesMetaInfo[pos_offset + 1] = 0;

            if (particleOffset[1] < 0) // 1 == y
                particlesMetaInfo[pos_offset + 1] = 0;

            int64_t adiosIndexVarId = *(params->adiosSpeciesIndexVarIds.begin());
            params->adiosSpeciesIndexVarIds.pop_front();
            ADIOS_CMD(adios_write_byid(params->adiosFileHandle, adiosIndexVarId, particlesMetaInfo));
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) writing particle index table for %1%") % AdiosFrameType::getName();
    }
};


} //namspace adios

} //namespace picongpu
