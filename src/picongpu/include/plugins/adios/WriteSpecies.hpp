/**
 * Copyright 2014 Rene Widera, Felix Schmitt
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

#include "RefWrapper.hpp"
#include <boost/type_traits.hpp>

#include "plugins/output/WriteSpeciesCommon.hpp"
#include "plugins/kernel/CopySpecies.kernel"
#include "mappings/kernel/AreaMapping.hpp"

#include "plugins/adios/writer/ParticleAttribute.hpp"
#include "compileTime/conversion/RemoveFromSeq.hpp"

namespace picongpu
{

namespace adios
{
using namespace PMacc;

namespace bmpl = boost::mpl;

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
    typedef typename FrameType::ValueTypeSeq ParticleAttributeList;
    typedef typename FrameType::MethodsList ParticleMethodsList;

    /* delete multiMask and localCellIdx in adios particle*/
    typedef bmpl::vector<multiMask,localCellIdx> TypesToDelete;
    typedef typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type ParticleCleanedAttributeList;

    /* add globalCellIdx for adios particle*/
    typedef typename MakeSeq<
            ParticleCleanedAttributeList, 
            globalCellIdx<globalCellIdx_pic>
    >::type ParticleNewAttributeList;

    typedef Frame<OperatorCreateVectorBox, ParticleNewAttributeList, ParticleMethodsList> AdiosFrameType;

    template<typename Space>
    HINLINE void operator()(RefWrapper<ThreadParams*> params,
                            const DomainInformation domInfo,
                            const Space particleOffset)
    {
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) write species: %1%") % AdiosFrameType::getName();
        DataConnector &dc = Environment<>::get().DataConnector();
        /* load particle without copy particle data to host */
        ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::getName(), true));

        /* count total number of particles on the device */
        log<picLog::INPUT_OUTPUT > ("ADIOS:  (begin) count particles: %1%") % AdiosFrameType::getName();
        uint64_cu totalNumParticles = 0;
        totalNumParticles = PMacc::CountParticles::countOnDevice < CORE + BORDER > (
                                                                                    *speciesTmp,
                                                                                    *(params.get()->cellDescription),
                                                                                    domInfo.localDomainOffset,
                                                                                    domInfo.domainSize);
        log<picLog::INPUT_OUTPUT > ("ADIOS:  ( end ) count particles: %1% = %2%") % AdiosFrameType::getName() % totalNumParticles;

        if (totalNumParticles > 0)
        {
            AdiosFrameType hostFrame;
            log<picLog::INPUT_OUTPUT > ("ADIOS:  (begin) malloc mapped memory: %1%") % AdiosFrameType::getName();

            /* malloc mapped memory */
            ForEach<typename AdiosFrameType::ValueTypeSeq, MallocMemory<void> > mallocMem;
            mallocMem(byRef(hostFrame), totalNumParticles);
            log<picLog::INPUT_OUTPUT > ("ADIOS:  ( end ) malloc mapped memory: %1%") % AdiosFrameType::getName();

            log<picLog::INPUT_OUTPUT > ("ADIOS:  (begin) get mapped memory device pointer: %1%") % AdiosFrameType::getName();
            /* load device pointer of mapped memory */
            AdiosFrameType deviceFrame;
            ForEach<typename AdiosFrameType::ValueTypeSeq, GetDevicePtr<void> > getDevicePtr;
            getDevicePtr(byRef(deviceFrame), byRef(hostFrame));
            log<picLog::INPUT_OUTPUT > ("ADIOS:  ( end ) get mapped memory device pointer: %1%") % AdiosFrameType::getName();

            log<picLog::INPUT_OUTPUT > ("ADIOS:  (begin) copy particle to host: %1%") % AdiosFrameType::getName();
            typedef bmpl::vector< typename GetPositionFilter<simDim>::type > usedFilters;
            typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
            MyParticleFilter filter;
            /* activeate filter pipeline if moving window is activated */
            filter.setStatus(MovingWindow::getInstance().isSlidingWindowActive());
            filter.setWindowPosition(domInfo.localDomainOffset, domInfo.domainSize);

            dim3 block(TILE_SIZE);
            DataSpace<simDim> superCells = speciesTmp->getParticlesBuffer().getSuperCellsCount();

            GridBuffer<int, DIM1> counterBuffer(DataSpace<DIM1>(1));
            AreaMapping < CORE + BORDER, MappingDesc > mapper(*(params.get()->cellDescription));

            __cudaKernel(copySpecies)
                (mapper.getGridDim(), block)
                (counterBuffer.getDeviceBuffer().getPointer(),
                 deviceFrame, speciesTmp->getDeviceParticlesBox(),
                 filter,
                 particleOffset, /*relative to data domain (not to physical domain)*/
                 mapper
                 );
            counterBuffer.deviceToHost();
            log<picLog::INPUT_OUTPUT > ("ADIOS:  ( end ) copy particle to host: %1%") % AdiosFrameType::getName();
            __getTransactionEvent().waitForFinished();
            log<picLog::INPUT_OUTPUT > ("ADIOS:  all events are finished: %1%") % AdiosFrameType::getName();
            /* this costs a little bit of time but adios writing is slower */
            assert((uint64_cu) counterBuffer.getHostBuffer().getDataBox()[0] == totalNumParticles);
            
            /* dump to adios file */        
            ForEach<typename AdiosFrameType::ValueTypeSeq, adios::ParticleAttribute<void> > writeToAdios;
            writeToAdios(params, byRef(hostFrame), totalNumParticles);
            
            /* free host memory */
            ForEach<typename AdiosFrameType::ValueTypeSeq, FreeMemory<void> > freeMem;
            freeMem(byRef(hostFrame));
            log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) writing species: %1%") % AdiosFrameType::getName();
        }

        /* write species counter table to adios file */
        log<picLog::INPUT_OUTPUT > ("ADIOS:  (begin) writing particle index table for %1%") % AdiosFrameType::getName();
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
            
            int64_t adiosIndexVarId = *(params.get()->adiosSpeciesIndexVarIds.begin());
            params.get()->adiosSpeciesIndexVarIds.pop_front();
            ADIOS_CMD(adios_write_byid(params.get()->adiosFileHandle, adiosIndexVarId, particlesMetaInfo));
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS:  ( end ) writing particle index table for %1%") % AdiosFrameType::getName(); 
    }
};


} //namspace adios

} //namespace picongpu
