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

#include "RefWrapper.hpp"
#include <boost/type_traits.hpp>

#include "plugins/hdf5/CopySpecies.kernel"
#include "mappings/kernel/AreaMapping.hpp"

#include "plugins/hdf5/writer/ParticleAttribute.hpp"

namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

using namespace DCollector;
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

/** Write calculated fields to HDF5 file.
 * 
 */
template< typename T >
struct WriteSpecies
{
public:

    typedef T ThisSpecies;
    typedef typename T::FrameType FrameType;
    typedef typename FrameType::ValueTypeSeq ParticleAttributeList;
    typedef typename FrameType::MethodsList ParticleMethodsList;

    /* at the moment some list opratations are not include in PMacc
     * this is the reason why we do so much magic
     */

    template<typename T_Key>
    struct isMultiMask
    {
        typedef typename GetKeyFromAlias<ParticleAttributeList, multiMask>::type Key;
        typedef boost::is_same< T_Key, Key> type;
    };

    typedef typename bmpl::remove_if< ParticleAttributeList, isMultiMask<bmpl::_> >::type NoMultiMask;

    template<typename T_Key>
    struct isLocalCellIdx
    {
        typedef typename GetKeyFromAlias<NoMultiMask, localCellIdx>::type Key;
        typedef boost::is_same< T_Key, Key> type;
    };

    typedef typename bmpl::remove_if< NoMultiMask, isLocalCellIdx<bmpl::_> >::type ParticleCleantAttributeList;

    typedef typename JoinVectors<ParticleCleantAttributeList, boost::mpl::vector<globalCellIdx<globalCellIdx_pic> > >::type ParticleNewAttributeList;

    typedef Frame<OperatorCreateVectorBox, ParticleNewAttributeList, ParticleMethodsList> Hdf5FrameType;

    HINLINE void operator()(RefWrapper<ThreadParams*> params,
                            std::string prefix,
                            const DataSpace<simDim> sim_offset,
                            const DataSpace<simDim> localOffset,
                            const DataSpace<simDim> localSize)
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
                                                                                    localOffset,
                                                                                    localSize);


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
            filter.setWindowPosition(localOffset, localOffset + localSize);

            dim3 block(TILE_SIZE);
            DataSpace<simDim> superCells = speciesTmp->getParticlesBuffer().getSuperCellsCount();

            GridBuffer<int, DIM1> counterBuffer(DataSpace<DIM1>(1));
            AreaMapping < CORE + BORDER, MappingDesc > mapper(*(params.get()->cellDescription));

            __cudaKernel(copySpecies)
                (mapper.getGridDim(), block)
                (counterBuffer.getDeviceBuffer().getPointer(),
                 deviceFrame, speciesTmp->getDeviceParticlesBox(),
                 filter,
                 sim_offset,
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
        writeToHdf5(params, byRef(hostFrame), prefix + FrameType::getName(), sim_offset, localSize, totalNumParticles);

        /*free host memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, FreeMemory<void> > freeMem;
        freeMem(byRef(hostFrame));
        log<picLog::INPUT_OUTPUT > ("HDF5: Finish write species: %1%") % Hdf5FrameType::getName();

    }
};


} //namspace hdf5

} //namepsace picongpu

