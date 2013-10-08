/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, René Widera
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
    HDINLINE void operator()(RefWrapper<ValueType> v1, const size_t size) const
    {
#ifndef __CUDA_ARCH__
        type* ptr;
        cudaHostAlloc(&ptr, size * sizeof (type), cudaHostAllocMapped);

        v1.get().getIdentifier(T_Type()) = VectorDataBox<type>(ptr);
#endif
    }
};

template<typename T_Type>
struct GetDevicePtr
{
    typedef typename T_Type::type type;

    template<typename ValueType >
    HDINLINE void operator()(RefWrapper<ValueType> dest, RefWrapper<ValueType> src) const
    {
#ifndef __CUDA_ARCH__
        type* ptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&ptr, src.get().getIdentifier(T_Type()).getPointer(), 0));
        dest.get().getIdentifier(T_Type()) =
            VectorDataBox<type>(ptr);
#endif
    }
};

template<typename T_Type>
struct FreeMemory
{

    template<typename ValueType >
    HDINLINE void operator()(RefWrapper<ValueType> value) const
    {
#ifndef __CUDA_ARCH__
        CUDA_CHECK(cudaFreeHost(value.get().getIdentifier(T_Type()).getPointer()));
#endif
    }
};

/** Write calculated fields to HDF5 file.
 * 
 */
template< typename T >
struct WriteSpecies
{
public:

    typedef T ThisSpecies;

    typedef typename T::FrameType::ValueTypeSeq ParticleAttributeList;
    typedef typename T::FrameType::MethodsList ParticleMethodsList;

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

    typedef Frame<CastToVectorBox, ParticleNewAttributeList, ParticleMethodsList> Hdf5FrameType;

    PMACC_NO_NVCC_HDWARNING
    HDINLINE void operator()(RefWrapper<ThreadParams*> tparam,
                             const DataSpace<simDim> sim_offset,
                             const DataSpace<simDim> localOffset,
                             const DataSpace<simDim> localSize)
    {
        this->operator_impl(tparam, sim_offset, localOffset, localSize);
    }

private:

    HINLINE void operator_impl(RefWrapper<ThreadParams*> params,
                               const DataSpace<simDim> sim_offset,
                               const DataSpace<simDim> localOffset,
                               const DataSpace<simDim> localSize)
    {

        DataConnector &dc = DataConnector::getInstance();
        /*load particle without copy particle data to host*/
        ThisSpecies* speciesTmp = &(dc.getData<ThisSpecies >(ThisSpecies::FrameType::CommunicationTag, true));

        // count total number of particles on the device
        uint64_cu totalNumParticles = 0;

        PMACC_AUTO(simBox, SubGrid<simDim>::getInstance().getSimulationBox());

        totalNumParticles = PMacc::CountParticles::countOnDevice < CORE + BORDER > (
                                                                                    *speciesTmp,
                                                                                    *(params.get()->cellDescription),
                                                                                    localOffset,
                                                                                    localSize);

        Hdf5FrameType hostFrame;
        /*malloc mapped memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, MallocMemory<void> > mallocMem;
        mallocMem(byRef(hostFrame), totalNumParticles);

        /*load device pointer of mapped memory*/
        Hdf5FrameType deviceFrame;
        ForEach<typename Hdf5FrameType::ValueTypeSeq, GetDevicePtr<void> > getDevicePtr;
        getDevicePtr(byRef(deviceFrame), byRef(hostFrame));


        typedef bmpl::vector< PositionFilter3D<> > usedFilters;
        typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
        MyParticleFilter filter;
        filter.setStatus(true); /*activeate filter pipline*/
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
             mapper
             );

        __getTransactionEvent().waitForFinished();

        /*free host memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, FreeMemory<void> > freeMem;
        freeMem(byRef(hostFrame));
    }
};


} //namspace hdf5

} //namepsace picongpu

