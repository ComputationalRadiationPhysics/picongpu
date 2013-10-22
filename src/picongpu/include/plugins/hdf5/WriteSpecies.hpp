/**
 * Copyright 2013 René Widera
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

#include "plugins/IPluginModule.hpp"
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

#include "plugins/hdf5/CopySpecies.kernel"
#include "mappings/kernel/AreaMapping.hpp"

#include "plugins/hdf5/writer/ParticleAttribute.hpp"
#include "compileTime/conversion/RemoveFromSeq.hpp"

namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

using namespace splash;
namespace bmpl = boost::mpl;

template<typename T_Type>
struct MallocMemory
{
    typedef typename T_Type::type type;

    template<typename ValueType >
    HINLINE void operator()(RefWrapper<ValueType> v1, const size_t size) const
    {
        type* ptr = NULL;
        if (size != 0)
        {
            CUDA_CHECK(cudaHostAlloc(&ptr, size * sizeof (type), cudaHostAllocMapped));
        }
        v1.get().getIdentifier(T_Type()) = VectorDataBox<type>(ptr);

    }
};

template<typename T_Type>
struct GetDevicePtr
{
    typedef typename T_Type::type type;

    template<typename ValueType >
    HINLINE void operator()(RefWrapper<ValueType> dest, RefWrapper<ValueType> src) const
    {
        type* ptr = NULL;
        type* srcPtr = src.get().getIdentifier(T_Type()).getPointer();
        if (srcPtr != NULL)
        {
            CUDA_CHECK(cudaHostGetDevicePointer(&ptr, srcPtr, 0));
        }
        dest.get().getIdentifier(T_Type()) =
            VectorDataBox<type>(ptr);
    }
};

template<typename T_Type>
struct FreeMemory
{
    typedef typename T_Type::type type;

    template<typename ValueType >
    HINLINE void operator()(RefWrapper<ValueType> value) const
    {
        type* ptr = value.get().getIdentifier(T_Type()).getPointer();
        if (ptr != NULL)
            CUDA_CHECK(cudaFreeHost(ptr));
    }
};

/*functor to create a pair for a MapTupel map*/
template<typename InType>
struct OperatorCreateVectorBox
{
    typedef
    bmpl::pair< InType,
    PMacc::VectorDataBox< typename InType::type > >
    type;
};

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

    typedef Frame<OperatorCreateVectorBox, ParticleNewAttributeList, ParticleMethodsList> Hdf5FrameType;

    template<typename Space>
    HINLINE void operator()(RefWrapper<ThreadParams*> params,
                            std::string prefix,
                            const DomainInformation domInfo,const Space particleOffset)
    {
        log<picLog::INPUT_OUTPUT > ("HDF5: write species: %1%") % Hdf5FrameType::getName();
        DataConnector &dc = DataConnector::getInstance();
        /*load particle without copy particle data to host*/
        ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::CommunicationTag, true));

        // count total number of particles on the device
        uint64_cu totalNumParticles = 0;

        PMACC_AUTO(simBox, SubGrid<simDim>::getInstance().getSimulationBox());

        log<picLog::INPUT_OUTPUT > ("HDF5: count particles: %1%") % Hdf5FrameType::getName();
        totalNumParticles = PMacc::CountParticles::countOnDevice < CORE + BORDER > (
                                                                                    *speciesTmp,
                                                                                    *(params.get()->cellDescription),
                                                                                    domInfo.localDomainOffset,
                                                                                    domInfo.domainSize);


        log<picLog::INPUT_OUTPUT > ("HDF5: Finish count particles: %1% = %2%") % Hdf5FrameType::getName() % totalNumParticles;
        Hdf5FrameType hostFrame;
        log<picLog::INPUT_OUTPUT > ("HDF5: malloc mapped memory: %1%") % Hdf5FrameType::getName();
        /*malloc mapped memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, MallocMemory<void> > mallocMem;
        mallocMem(byRef(hostFrame), totalNumParticles);
        log<picLog::INPUT_OUTPUT > ("HDF5: Finish malloc mapped memory: %1%") % Hdf5FrameType::getName();

        if (totalNumParticles != 0)
        {

            log<picLog::INPUT_OUTPUT > ("HDF5: get mapped memory device pointer: %1%") % Hdf5FrameType::getName();
            /*load device pointer of mapped memory*/
            Hdf5FrameType deviceFrame;
            ForEach<typename Hdf5FrameType::ValueTypeSeq, GetDevicePtr<void> > getDevicePtr;
            getDevicePtr(byRef(deviceFrame), byRef(hostFrame));
            log<picLog::INPUT_OUTPUT > ("HDF5: Finish get mapped memory device pointer: %1%") % Hdf5FrameType::getName();

            log<picLog::INPUT_OUTPUT > ("HDF5: copy particle to host: %1%") % Hdf5FrameType::getName();
            typedef bmpl::vector< PositionFilter3D<> > usedFilters;
            typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
            MyParticleFilter filter;
            /*activeate filter pipline if moving window is activated*/
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
            log<picLog::INPUT_OUTPUT > ("HDF5: memcpy particle counter to host: %1%") % Hdf5FrameType::getName();
            __getTransactionEvent().waitForFinished();
            log<picLog::INPUT_OUTPUT > ("HDF5: all events are finish: %1%") % Hdf5FrameType::getName();
            /*this cost a little bit of time but hdf5 writing is slower^^*/
            assert((uint64_cu) counterBuffer.getHostBuffer().getDataBox()[0] == totalNumParticles);
        }
        /*dump to hdf5 file*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, hdf5::ParticleAttribute<void> > writeToHdf5;
        writeToHdf5(params, byRef(hostFrame), prefix + FrameType::getName(), domInfo, totalNumParticles);

        /*free host memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, FreeMemory<void> > freeMem;
        freeMem(byRef(hostFrame));
        log<picLog::INPUT_OUTPUT > ("HDF5: Finish write species: %1%") % Hdf5FrameType::getName();

    }
};


} //namspace hdf5

} //namespace picongpu

