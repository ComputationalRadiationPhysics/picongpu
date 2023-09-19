/* Copyright 2013-2023 Rene Widera, Erik Zenker
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

#include "pmacc/kernel/atomic.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/memory/shared/Allocate.hpp"
#include "pmacc/particles/algorithm/ForEach.hpp"
#include "pmacc/particles/memory/dataTypes/FramePointer.hpp"
#include "pmacc/particles/particleFilter/FilterFactory.hpp"
#include "pmacc/particles/particleFilter/PositionFilter.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/types.hpp"


namespace pmacc
{
    /* count particles
     *
     * it is allowed to call this kernel on frames with holes (without calling fillAllGAps before)
     */
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
        template<typename T_PBox, typename T_Filter, typename T_Mapping, typename T_ParticleFilter, typename T_Worker>
        DINLINE void operator()(
            T_Worker const& worker,
            T_PBox pb,
            uint64_cu* gCounter,
            T_Filter filter,
            T_Mapping const mapper,
            T_ParticleFilter parFilter) const
        {
            constexpr uint32_t dim = T_Mapping::Dim;

            PMACC_SMEM(worker, counter, int);
            DataSpace<dim> const superCellIdx(
                mapper.getSuperCellIndex(DataSpace<dim>(cupla::blockIdx(worker.getAcc()))));

            auto onlyMaster = lockstep::makeMaster(worker);

            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, pb, superCellIdx);

            onlyMaster([&]() { counter = 0; });

            worker.sync();

            // end kernel if we have no particles
            if(!forEachParticle.hasParticles())
                return;
            filter.setSuperCellPosition((superCellIdx - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());

            auto accParFilter = parFilter(worker, superCellIdx - mapper.getGuardingSuperCells());

            forEachParticle(
                [&counter, &filter, &accParFilter](auto const& lockstepWorker, auto& particle)
                {
                    bool const useParticle = filter(particle);
                    if(useParticle)
                    {
                        if(accParFilter(lockstepWorker, particle))
                            kernel::atomicAllInc(lockstepWorker, &counter, ::alpaka::hierarchy::Threads{});
                    }
                });

            worker.sync();

            onlyMaster(
                [&]() {
                    cupla::atomicAdd(
                        worker.getAcc(),
                        gCounter,
                        static_cast<uint64_cu>(counter),
                        ::alpaka::hierarchy::Blocks{});
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

            auto const mapper = makeAreaMapper<AREA>(cellDescription);
            auto workerCfg = lockstep::makeWorkerCfg<PBuffer::FrameType::frameSize>();

            PMACC_LOCKSTEP_KERNEL(KernelCountParticles{}, workerCfg)
            (mapper.getGridDim())(
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
            using usedFilters = mp_list<typename GetPositionFilter<Space::Dim>::type>;
            using MyParticleFilter = typename FilterFactory<usedFilters>::FilterType;
            MyParticleFilter filter;
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
