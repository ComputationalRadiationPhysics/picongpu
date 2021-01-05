/* Copyright 2013-2021 Rene Widera, Erik Zenker
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/particles/memory/dataTypes/FramePointer.hpp"

#include "pmacc/particles/particleFilter/FilterFactory.hpp"
#include "pmacc/particles/particleFilter/PositionFilter.hpp"
#include "pmacc/nvidia/atomic.hpp"
#include "pmacc/memory/shared/Allocate.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/mappings/threads/ForEachIdx.hpp"
#include "pmacc/mappings/threads/IdxConfig.hpp"


namespace pmacc
{
    /* count particles
     *
     * it is allowed to call this kernel on frames with holes (without calling fillAllGAps before)
     *
     * @tparam T_numWorkers number of workers
     */
    template<uint32_t T_numWorkers>
    struct KernelCountParticles
    {
        /** count particles
         *
         * @tparam T_PBox pmacc::ParticlesBox, particle box type
         * @tparam T_Filter functor to filter particles
         * @tparam T_Mapping supercell mapper functor type
         * @tparam T_ParticleFilter pmacc::filter::Interface, type of the particle filter
         * @tparam T_Acc type of the alpaka accelerator
         *
         * @param pb particle memory
         * @param gCounter pointer for the result
         * @param filter functor to filter particles those should be counted
         * @param mapper functor to map a block to a supercell
         * @param parFilter particle filter method, the working domain for the filter is supercells
         */
        template<typename T_PBox, typename T_Filter, typename T_Mapping, typename T_ParticleFilter, typename T_Acc>
        DINLINE void operator()(
            T_Acc const& acc,
            T_PBox pb,
            uint64_cu* gCounter,
            T_Filter filter,
            T_Mapping const mapper,
            T_ParticleFilter parFilter) const
        {
            using namespace mappings::threads;

            using Frame = typename T_PBox::FrameType;
            using FramePtr = typename T_PBox::FramePtr;
            constexpr uint32_t dim = T_Mapping::Dim;
            constexpr uint32_t frameSize = math::CT::volume<typename Frame::SuperCellSize>::type::value;
            constexpr uint32_t numWorkers = T_numWorkers;

            PMACC_SMEM(acc, frame, FramePtr);
            PMACC_SMEM(acc, counter, int);
            PMACC_SMEM(acc, particlesInSuperCell, lcellId_t);

            using SuperCellSize = typename T_Mapping::SuperCellSize;

            DataSpace<dim> const threadIndex(cupla::threadIdx(acc));
            uint32_t const workerIdx
                = static_cast<uint32_t>(DataSpaceOperations<dim>::template map<SuperCellSize>(threadIndex));

            DataSpace<dim> const superCellIdx(mapper.getSuperCellIndex(DataSpace<dim>(cupla::blockIdx(acc))));

            ForEachIdx<IdxConfig<1, numWorkers>> onlyMaster{workerIdx};

            onlyMaster([&](uint32_t const, uint32_t const) {
                frame = pb.getLastFrame(superCellIdx);
                particlesInSuperCell = pb.getSuperCell(superCellIdx).getSizeLastFrame();
                counter = 0;
            });

            cupla::__syncthreads(acc);

            if(!frame.isValid())
                return; // end kernel if we have no frames
            filter.setSuperCellPosition((superCellIdx - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());

            auto accParFilter
                = parFilter(acc, superCellIdx - mapper.getGuardingSuperCells(), WorkerCfg<numWorkers>{workerIdx});

            ForEachIdx<IdxConfig<frameSize, numWorkers>> forEachParticle(workerIdx);

            while(frame.isValid())
            {
                forEachParticle([&](uint32_t const linearIdx, uint32_t const idx) {
                    if(linearIdx < particlesInSuperCell)
                    {
                        bool const useParticle = filter(*frame, linearIdx);
                        if(useParticle)
                        {
                            auto parSrc = (frame[linearIdx]);
                            if(accParFilter(acc, parSrc))
                                nvidia::atomicAllInc(acc, &counter, ::alpaka::hierarchy::Threads{});
                        }
                    }
                });

                cupla::__syncthreads(acc);

                onlyMaster([&](uint32_t const, uint32_t const) {
                    frame = pb.getPreviousFrame(frame);
                    particlesInSuperCell = frameSize;
                });

                cupla::__syncthreads(acc);
            }

            onlyMaster([&](uint32_t const, uint32_t const) {
                cupla::atomicAdd(acc, gCounter, static_cast<uint64_cu>(counter), ::alpaka::hierarchy::Blocks{});
            });
        }
    };

    struct CountParticles
    {
        /** Get particle count
         *
         * @tparam AREA area were particles are counted (CORE, BORDER, GUARD)
         *
         * @param buffer source particle buffer
         * @param cellDescription instance of MappingDesction
         * @param filter filter instance which must inharid from PositionFilter
         * @param parFilter particle filter method, must fulfill the interface of pmacc::filter::Interface
         *                  The working domain for the filter is supercells.
         * @return number of particles in defined area
         */
        template<uint32_t AREA, class PBuffer, class Filter, class CellDesc, typename T_ParticleFilter>
        static uint64_cu countOnDevice(
            PBuffer& buffer,
            CellDesc cellDescription,
            Filter filter,
            T_ParticleFilter& parFilter)
        {
            GridBuffer<uint64_cu, DIM1> counter(DataSpace<DIM1>(1));

            AreaMapping<AREA, CellDesc> mapper(cellDescription);
            constexpr uint32_t numWorkers
                = traits::GetNumWorkers<math::CT::volume<typename CellDesc::SuperCellSize>::type::value>::value;

            PMACC_KERNEL(KernelCountParticles<numWorkers>{})
            (mapper.getGridDim(), numWorkers)(
                buffer.getDeviceParticlesBox(),
                counter.getDeviceBuffer().getBasePointer(),
                filter,
                mapper,
                parFilter);

            counter.deviceToHost();
            return *(counter.getHostBuffer().getDataBox());
        }

        /** Get particle count
         *
         * @param buffer source particle buffer
         * @param cellDescription instance of MappingDesction
         * @param filter filter instance which must inharid from PositionFilter
         * @param parFilter particle filter method, must fulfill the interface of pmacc::filter::Interface
         *                  The working domain for the filter is supercells.
         * @return number of particles in defined area
         */
        template<class PBuffer, class Filter, class CellDesc, typename T_ParticleFilter>
        static uint64_cu countOnDevice(
            PBuffer& buffer,
            CellDesc cellDescription,
            Filter filter,
            T_ParticleFilter& parFilter)
        {
            return pmacc::CountParticles::countOnDevice<CORE + BORDER + GUARD>(
                buffer,
                cellDescription,
                filter,
                parFilter);
        }

        /** Get particle count
         *
         * @tparam AREA area were particles are counted (CORE, BORDER, GUARD)
         *
         * @param buffer source particle buffer
         * @param cellDescription instance of MappingDesction
         * @param origin local cell position (can be negative)
         * @param size local size in cells for checked volume
         * @param parFilter particle filter method, must fulfill the interface of pmacc::filter::Interface
         *                  The working domain for the filter is supercells.
         * @return number of particles in defined area
         */
        template<uint32_t AREA, class PBuffer, class CellDesc, class Space, typename T_ParticleFilter>
        static uint64_cu countOnDevice(
            PBuffer& buffer,
            CellDesc cellDescription,
            const Space& origin,
            const Space& size,
            T_ParticleFilter& parFilter)
        {
            typedef bmpl::vector<typename GetPositionFilter<Space::Dim>::type> usedFilters;
            typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
            MyParticleFilter filter;
            filter.setStatus(true); /*activeate filter pipline*/
            filter.setWindowPosition(origin, size);
            return pmacc::CountParticles::countOnDevice<AREA>(buffer, cellDescription, filter, parFilter);
        }

        /** Get particle count
         *
         * @param buffer source particle buffer
         * @param cellDescription instance of MappingDesction
         * @param origin local cell position (can be negative)
         * @param size local size in cells for checked volume
         * @param parFilter particle filter method, must fulfill the interface of pmacc::filter::Interface
         *                  The working domain for the filter is supercells.
         * @return number of particles in defined area
         */
        template<class PBuffer, class Filter, class CellDesc, class Space, typename T_ParticleFilter>
        static uint64_cu countOnDevice(
            PBuffer& buffer,
            CellDesc cellDescription,
            const Space& origin,
            const Space& size,
            T_ParticleFilter& parFilter)
        {
            return pmacc::CountParticles::countOnDevice<CORE + BORDER + GUARD>(
                buffer,
                cellDescription,
                origin,
                size,
                parFilter);
        }
    };

} // namespace pmacc
