/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "types.h"
#include "memory/buffers/GridBuffer.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"

namespace PMacc
{

/* count particles in an area
 * is not optimized, it checks any partcile position if its realy a particle
 */
template<class PBox, class Filter, class Mapping>
__global__ void kernelCountParticles(PBox pb,
                                     uint64_cu* gCounter,
                                     Filter filter,
                                     Mapping mapper)
{

    typedef typename PBox::FrameType FRAME;
    const uint32_t Dim = Mapping::Dim;

    __shared__ FRAME *frame;
    __shared__ bool isValid;
    __shared__ int counter;
    __shared__ lcellId_t particlesInSuperCell;


    __syncthreads(); /*wait that all shared memory is initialised*/

    typedef typename Mapping::SuperCellSize SuperCellSize;

    const DataSpace<Dim > threadIndex(threadIdx);
    const int linearThreadIdx = DataSpaceOperations<Dim>::template map<SuperCellSize > (threadIndex);
    const DataSpace<Dim> superCellIdx(mapper.getSuperCellIndex(DataSpace<Dim > (blockIdx)));

    if (linearThreadIdx == 0)
    {
        frame = &(pb.getLastFrame(superCellIdx, isValid));
        particlesInSuperCell = pb.getSuperCell(superCellIdx).getSizeLastFrame();
        counter = 0;
    }
    __syncthreads();
    if (!isValid)
        return; //end kernel if we have no frames
    filter.setSuperCellPosition((superCellIdx - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());
    while (isValid)
    {
        if (linearThreadIdx < particlesInSuperCell)
        {
            if (filter(*frame, linearThreadIdx))
                atomicAdd(&counter, 1);
        }
        __syncthreads();
        if (linearThreadIdx == 0)
        {
            frame = &(pb.getPreviousFrame(*frame, isValid));
            particlesInSuperCell = math::CT::volume<SuperCellSize>::type::value;
        }
        __syncthreads();
    }

    __syncthreads();
    if (linearThreadIdx == 0)
    {
        atomicAdd(gCounter, (uint64_cu) counter);
    }
}

struct CountParticles
{

    /** Get particle count
     *
     * @tparam AREA area were particles are counted (CORE, BORDER, GUARD)
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param filter filter instance which must inharid from PositionFilter
     * @return number of particles in defined area
     */
    template<uint32_t AREA, class PBuffer, class Filter, class CellDesc>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, Filter filter)
    {
        GridBuffer<uint64_cu, DIM1> counter(DataSpace<DIM1>(1));

        dim3 block(CellDesc::SuperCellSize::toRT().toDim3());

        AreaMapping<AREA, CellDesc> mapper(cellDescription);

        __cudaKernel(kernelCountParticles)
            (mapper.getGridDim(), block)
            (buffer.getDeviceParticlesBox(),
             counter.getDeviceBuffer().getBasePointer(),
             filter,
             mapper);

        counter.deviceToHost();
        return *(counter.getHostBuffer().getDataBox());
    }

    /** Get particle count
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param filter filter instance which must inharid from PositionFilter
     * @return number of particles in defined area
     */
    template< class PBuffer, class Filter, class CellDesc>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, Filter filter)
    {
        return PMacc::CountParticles::countOnDevice < CORE + BORDER + GUARD > (buffer, cellDescription, filter);
    }

    /** Get particle count
     *
     * @tparam AREA area were particles are counted (CORE, BORDER, GUARD)
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param origin local cell position (can be negative)
     * @param size local size in cells for checked volume
     * @return number of particles in defined area
     */
    template<uint32_t AREA, class PBuffer, class CellDesc, class Space>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, const Space& origin, const Space& size)
    {
        typedef bmpl::vector< typename GetPositionFilter<Space::Dim>::type > usedFilters;
        typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
        MyParticleFilter filter;
        filter.setStatus(true); /*activeate filter pipline*/
        filter.setWindowPosition(origin, size);
        return PMacc::CountParticles::countOnDevice<AREA>(buffer, cellDescription, filter);
    }

    /** Get particle count
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param origin local cell position (can be negative)
     * @param size local size in cells for checked volume
     * @return number of particles in defined area
     */
    template< class PBuffer, class Filter, class CellDesc, class Space>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, const Space& origin, const Space& size)
    {
        return PMacc::CountParticles::countOnDevice < CORE + BORDER + GUARD > (buffer, cellDescription, origin, size);
    }

};

} //namespace PMacc
