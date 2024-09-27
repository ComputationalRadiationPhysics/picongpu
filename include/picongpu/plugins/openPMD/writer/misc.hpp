/* Copyright 2014-2023 Rene Widera, Felix Schmitt, Axel Huebl,
 *                     Alexander Grund, Franz Poeschel
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/eventSystem/events/kernelEvents.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/RangeMapping.hpp>
#include <pmacc/particles/operations/ConcatListOfFrames.hpp>
#include <pmacc/particles/particleFilter/FilterFactory.hpp>
#include <pmacc/particles/particleFilter/PositionFilter.hpp>

#include <boost/mpl/placeholders.hpp>

#include <algorithm>
#include <type_traits> // std::remove_reference_t

namespace picongpu
{
    namespace openPMD
    {
        struct KernelSuperCellHistogram
        {
            template<
                typename T_Mapping,
                typename T_Worker,
                typename T_ParBox,
                typename T_Filter,
                typename T_PositionFilter>
            DINLINE void operator()(
                T_Worker const& worker,
                T_ParBox pb,
                uint32_t* particlesPerSuperCell,
                T_Filter filter,
                T_PositionFilter posFilter,
                T_Mapping mapper) const
            {
                DataSpace<simDim> const superCellIdxND(mapper.getSuperCellIndex(worker.blockDomIdxND()));

                PMACC_SMEM(worker, numParticlesInBlock, uint32_t);

                auto onlyMaster = lockstep::makeMaster(worker);
                onlyMaster([&]() { numParticlesInBlock = 0u; });

                worker.sync();

                auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, pb, superCellIdxND);
                if(forEachParticle.hasParticles())
                {
                    uint32_t parCount = 0u;
                    auto accFilter = filter(worker, superCellIdxND - mapper.getGuardingSuperCells());
                    posFilter.setSuperCellPosition(
                        (superCellIdxND - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());
                    forEachParticle(
                        [&accFilter, &posFilter, &worker, &parCount](auto const& lockstepWorker, auto& particle)
                        {
                            if(accFilter(lockstepWorker, particle) && posFilter(particle))
                            {
                                parCount++;
                            }
                        });

                    worker.sync();
                    alpaka::atomicAdd(worker.getAcc(), &numParticlesInBlock, parCount, alpaka::hierarchy::Threads{});
                }

                worker.sync();

                auto superCellIdx = mapper.superCellIdx(worker.blockDomIdxND());
                onlyMaster([&]() { particlesPerSuperCell[superCellIdx] = numParticlesInBlock; });
            }
        };

        struct ChunkDescription
        {
            //! index of the first supercell with particles
            size_t beginSupercellIdx = 0u;
            //! index of the supercell behind the last usable supercell
            size_t endSupercellIdx = 0u;
            //! number of particles within the range of supercells
            size_t numberOfParticles = 0u;
        };

        struct ParticleIoChunkInfo
        {
            std::vector<ChunkDescription> ranges;
            // size of the largest chunk (in number of particles)
            size_t largestChunk = 0u;
            // total number of particles on the device
            size_t totalNumParticles = 0u;

            //! updates
            void emplace(size_t beginSupercellIdx, size_t endSupercellIdx, size_t numberOfParticles)
            {
                ranges.push_back({beginSupercellIdx, endSupercellIdx, numberOfParticles});
                largestChunk = std::max(largestChunk, numberOfParticles);
            }
        };

        /** chunk information required to copy particles to OpenPMD without violating the maximum IO chunk size given
         * by the user
         *
         * @attention In case a single supercell contains more particles than fitting a buffer with the maximum size
         * given via CLI parameter 'particleIOChunkSize' the maximum will be increased automatically to ensure all
         * particles can be dumped.
         *
         * @param params openPMD plugin parameter (collection class contains nearly every important data for IO)
         * @param speciedPtr shared pointer to the species which should be dumped
         * @param particleFilter particle filter to decide which particle is taken into account based on particle
         * attribute values
         * @param particleSizeInByte Size of a single particle (in byte) with the attributes used for IO.
         *                           The size can be different compared to the particle layout used within the
         *                           simulation.
         * @return chunk information
         */
        template<typename T_IOParameters, typename T_SpeciesPtr, typename T_ParticleFilter>
        inline ParticleIoChunkInfo createSupercellRangeChunks(
            T_IOParameters* params,
            T_SpeciesPtr& speciedPtr,
            T_ParticleFilter particleFilter,
            size_t particleSizeInByte)
        {
            auto const areaMapper = makeAreaMapper<CORE + BORDER>(*(params->cellDescription));
            auto rangeMapper = makeRangeMapper(areaMapper);

            // buffer accessible from device and host to store the number of particles within each supercell.
            auto superCellHistogram = alpaka::allocMappedBufIfSupported<uint32_t, MemIdxType>(
                manager::Device<HostDevice>::get().current(),
                manager::Device<ComputeDevice>::get().getPlatform(),
                MemSpace<DIM1>(rangeMapper.size()).toAlpakaMemVec());
            uint32_t* histData = alpaka::getPtrNative(superCellHistogram);

            using UsedPositionFilters = mp_list<typename GetPositionFilter<simDim>::type>;
            using MyParticleFilter = typename FilterFactory<UsedPositionFilters>::FilterType;
            MyParticleFilter posFilter;
            posFilter.setWindowPosition(params->localWindowToDomainOffset, params->window.localDimensions.size);

            PMACC_LOCKSTEP_KERNEL(KernelSuperCellHistogram{})
                .config(rangeMapper.getGridDim(), *speciedPtr)(
                    speciedPtr->getDeviceParticlesBox(),
                    histData,
                    particleFilter,
                    posFilter,
                    rangeMapper);
            eventSystem::getTransactionEvent().waitForFinished();

            // max buffer size in bytes, value is provided as CLI parameter by the user
            size_t maxBufferSize = params->particleIOChunkSize * 1024u * 1024u;

            log<picLog::INPUT_OUTPUT>("openPMD: Max particle chunk buffer size set by user: %1% MiB")
                % params->particleIOChunkSize;

            size_t maxAllowedParticlesPerChunk = maxBufferSize / particleSizeInByte;
            size_t currentChunkSize = 0u;

            ParticleIoChunkInfo particleIoChunkInfo;

            size_t lastBeginSupercellIdx = 0u;
            // @attention the loop includes the last index, therefore a range check within the body is required
            for(size_t supercellIdx = 0u; supercellIdx <= rangeMapper.size(); ++supercellIdx)
            {
                if(supercellIdx < rangeMapper.size())
                {
                    if(histData[supercellIdx] > maxAllowedParticlesPerChunk)
                    {
                        size_t increasedParticleChunkSize = histData[supercellIdx];
                        std::cerr << "Warning: more particles per supercell than we can fit into "
                                  << (maxAllowedParticlesPerChunk * particleSizeInByte)
                                  << "byte, chunk limit is increased to "
                                  << (increasedParticleChunkSize * particleSizeInByte) << " bytes." << std::endl;
                        maxAllowedParticlesPerChunk = increasedParticleChunkSize;
                    }
                    size_t currentDataSizeBytes = currentChunkSize + histData[supercellIdx];
                    if(currentDataSizeBytes > maxAllowedParticlesPerChunk)
                    {
                        particleIoChunkInfo.emplace(lastBeginSupercellIdx, supercellIdx, currentChunkSize);
                        // start the next chunk
                        currentChunkSize = histData[supercellIdx];
                        lastBeginSupercellIdx = supercellIdx;
                    }
                    else
                    {
                        currentChunkSize += histData[supercellIdx];
                    }
                    particleIoChunkInfo.totalNumParticles += histData[supercellIdx];
                }
                else if(currentChunkSize != 0u)
                {
                    // include all not processed particles within a last chunk
                    particleIoChunkInfo.emplace(lastBeginSupercellIdx, supercellIdx, currentChunkSize);
                }
            }

            return particleIoChunkInfo;
        }
    } // namespace openPMD
} // namespace picongpu
