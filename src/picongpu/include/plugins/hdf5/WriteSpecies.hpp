/**
 * Copyright 2013-2014 Rene Widera, Felix Schmitt
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
#include "plugins/hdf5/HDF5Writer.def"

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

#include "plugins/hdf5/writer/ParticleAttribute.hpp"
#include "compileTime/conversion/RemoveFromSeq.hpp"
#include "particles/ParticleDescription.hpp"

namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

// = ColTypeUInt64_5Array
TYPE_ARRAY(UInt64_5, H5T_INTEL_U64, uint64_t, 5);

using namespace splash;
namespace bmpl = boost::mpl;

/** Write copy particle to host memory and dump to HDF5 file
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

    /* delete multiMask and localCellIdx in hdf5 particle*/
    typedef bmpl::vector<multiMask,localCellIdx> TypesToDelete;
    typedef typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type ParticleCleanedAttributeList;

    /* add globalCellIdx for hdf5 particle*/
    typedef typename MakeSeq<
            ParticleCleanedAttributeList, 
            globalCellIdx<globalCellIdx_pic>
    >::type ParticleNewAttributeList;

    typedef Frame<OperatorCreateVectorBox, 
        ParticleDescription<typename FrameType::Name,ParticleNewAttributeList, ParticleMethodsList> 
    > Hdf5FrameType;

    template<typename Space>
    HINLINE void operator()(RefWrapper<ThreadParams*> params,
                            std::string subGroup,
                            const DomainInformation domInfo,
                            const Space particleOffset)
    {
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) write species: %1%") % Hdf5FrameType::getName();
        DataConnector &dc = Environment<>::get().DataConnector();
        /* load particle without copy particle data to host */
        ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::getName(), true));

        /* count total number of particles on the device */
        uint64_cu totalNumParticles = 0;

        PMACC_AUTO(simBox, Environment<simDim>::get().SubGrid().getSimulationBox());

        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) count particles: %1%") % Hdf5FrameType::getName();
        totalNumParticles = PMacc::CountParticles::countOnDevice < CORE + BORDER > (
                                                                                    *speciesTmp,
                                                                                    *(params.get()->cellDescription),
                                                                                    domInfo.localDomainOffset,
                                                                                    domInfo.domainSize);


        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) count particles: %1% = %2%") % Hdf5FrameType::getName() % totalNumParticles;
        Hdf5FrameType hostFrame;
        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) malloc mapped memory: %1%") % Hdf5FrameType::getName();
        /*malloc mapped memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, MallocMemory<void> > mallocMem;
        mallocMem(byRef(hostFrame), totalNumParticles);
        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) malloc mapped memory: %1%") % Hdf5FrameType::getName();

        if (totalNumParticles != 0)
        {

            log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) get mapped memory device pointer: %1%") % Hdf5FrameType::getName();
            /*load device pointer of mapped memory*/
            Hdf5FrameType deviceFrame;
            ForEach<typename Hdf5FrameType::ValueTypeSeq, GetDevicePtr<void> > getDevicePtr;
            getDevicePtr(byRef(deviceFrame), byRef(hostFrame));
            log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) get mapped memory device pointer: %1%") % Hdf5FrameType::getName();

            log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) copy particle to host: %1%") % Hdf5FrameType::getName();
            typedef bmpl::vector< typename GetPositionFilter<simDim>::type > usedFilters;
            typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
            MyParticleFilter filter;
            /* activate filter pipeline if moving window is activated */
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
            log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) copy particle to host: %1%") % Hdf5FrameType::getName();
            __getTransactionEvent().waitForFinished();
            log<picLog::INPUT_OUTPUT > ("HDF5:  all events are finished: %1%") % Hdf5FrameType::getName();
            /*this cost a little bit of time but hdf5 writing is slower^^*/
            assert((uint64_cu) counterBuffer.getHostBuffer().getDataBox()[0] == totalNumParticles);
        }
        /*dump to hdf5 file*/        
        ForEach<typename Hdf5FrameType::ValueTypeSeq, hdf5::ParticleAttribute<void> > writeToHdf5;
        writeToHdf5(params, byRef(hostFrame), std::string("particles/") + FrameType::getName() + std::string("/") + subGroup,
                domInfo, totalNumParticles);
        
        /* write meta attributes for species */
        writeMetaAttributes(params.get());

        /*write species counter table to hdf5 file*/
        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) writing particle index table for %1%") % Hdf5FrameType::getName();
        {
            ColTypeUInt64_5Array ctUInt64_5;
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

            params.get()->dataCollector->write(
                params.get()->currentStep,
                Dimensions(gc.getGlobalSize(), 1, 1),
                Dimensions(gc.getGlobalRank(), 0, 0),
                ctUInt64_5, 1,
                Dimensions(1, 1, 1),
                (std::string("particles/") + FrameType::getName() + std::string("/") +
                    subGroup + std::string("/particles_info")).c_str(),
                particlesMetaInfo);
        }
        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) writing particle index table for %1%") % Hdf5FrameType::getName();
        
        /*free host memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, FreeMemory<void> > freeMem;
        freeMem(byRef(hostFrame));
        log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing species: %1%") % Hdf5FrameType::getName();
    }
    
private:
     
    /**
     * Writes additional meta-attributes directly to species group
     * 
     * @param params thread parameters
     */
    static void writeMetaAttributes(ThreadParams* params)
    {
        typedef typename PICToSplash<float_64>::type SplashFloat64Type;
        
        SplashFloat64Type splashType;
        
        const std::string groupName = std::string("particles/") + FrameType::getName();
        
        const float_64 charge = (float_64)FrameType::getCharge(1.0);
        params->dataCollector->writeAttribute(params->currentStep,
                splashType, groupName.c_str(), "charge", &charge);
        
        const float_64 mass = (float_64)FrameType::getMass(1.0);
        params->dataCollector->writeAttribute(params->currentStep,
                splashType, groupName.c_str(), "mass", &mass);
    }
};


} //namspace hdf5

} //namespace picongpu
