/* Copyright 2013-2024 Rene Widera, Felix Schmitt, Axel Huebl, Franz Poeschel
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

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/defines.hpp"
#    include "picongpu/plugins/ISimulationPlugin.hpp"
#    include "picongpu/plugins/openPMD/openPMDWriter.def"
#    include "picongpu/plugins/openPMD/restart/LoadParticleAttributesFromOpenPMD.hpp"
#    include "picongpu/plugins/output/WriteSpeciesCommon.hpp"

#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/meta/conversion/MakeSeq.hpp>
#    include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#    include <pmacc/particles/ParticleDescription.hpp>
#    include <pmacc/particles/operations/splitIntoListOfFrames.kernel>

#    include <boost/mpl/placeholders.hpp>

#    include <algorithm>
#    include <cassert>
#    include <stdexcept>
#    include <utility>

#    include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;

        struct RedistributeFilteredParticlesKernel
        {
            template<typename T_Worker, typename ValueType, typename FilterType, typename RemapType>
            HDINLINE void operator()(
                T_Worker const& worker,
                ValueType* dataPtr,
                FilterType&& filter,
                RemapType&& remap,
                MemIdxType const size,
                char const filterRemove) const
            {
                constexpr uint32_t blockDomSize = T_Worker::blockDomSize();
                auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;

                ValueType* s_mem = ::alpaka::getDynSharedMem<ValueType>(worker.getAcc());

                // grid-strided loop over the chunked data
                for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
                {
                    auto dataBlockOffset = dataBlock * blockDomSize;
                    auto forEach = pmacc::lockstep::makeForEach(worker);
                    // read
                    forEach(
                        [&](uint32_t const inBlockIdx)
                        {
                            auto idx = dataBlockOffset + inBlockIdx;
                            if(idx < size)
                            {
                                s_mem[inBlockIdx] = dataPtr[idx];
                            }
                        });
                    worker.sync();

                    // write
                    forEach(
                        [&](uint32_t const inBlockIdx)
                        {
                            auto idx = dataBlockOffset + inBlockIdx;
                            if(idx < size && filter[idx] != filterRemove)
                            {
                                dataPtr[remap[idx]] = s_mem[inBlockIdx];
                            }
                        });

                    // The next Iteration of the outer for loop does not depend on the data overwritten until now.
                    // Filtering moves data only to lower indexes and each of the outer loop's iterations deals with a
                    // contiguous chunk of data.
                }
            }
        };

        template<typename T_Identifier>
        struct RedistributeFilteredParticles

        {
            template<typename FrameType, typename FilterType, typename RemapType>
            HINLINE void operator()(
                FrameType& frame,
                FilterType const& filter,
                RemapType const& remap,
                uint64_t const numParticlesCurrentBatch,
                char const filterRemove)
            {
                using Identifier = T_Identifier;
                using ValueType = typename pmacc::traits::Resolve<Identifier>::type::type;
                using ComponentType = typename GetComponentsType<ValueType>::type;


                constexpr uint32_t blockDomSize = decltype(lockstep::makeBlockCfg<DIM1>())::blockDomSize();
                constexpr size_t requiredSharedMemBytes = blockDomSize * sizeof(ValueType);

                ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer();

                PMACC_LOCKSTEP_KERNEL(RedistributeFilteredParticlesKernel{})
                    .configSMem<DIM1>(pmacc::math::Vector{numParticlesCurrentBatch}, requiredSharedMemBytes)(
                        dataPtr,
                        alpaka::getPtrNative(filter),
                        alpaka::getPtrNative(remap),
                        numParticlesCurrentBatch,
                        filterRemove);
            }
        };

        struct KernelFilterParticles
        {
            template<typename T_Worker, typename FrameType, typename FilterType>
            DINLINE void operator()(
                T_Worker const& worker,
                FrameType&& loadedData,
                MemIdxType size,
                DataSpace<simDim> patchTotalOffset,
                DataSpace<simDim> patchUpperCorner,
                char const filterKeep,
                char const filterRemove,
                FilterType&& filterOut) const
            {
                // DataSpace<1> particleIndex = worker.blockDomIdxND();
                constexpr uint32_t blockDomSize = T_Worker::blockDomSize();
                auto numDataBlocks = (size + blockDomSize - 1u) / blockDomSize;
                auto position_ = loadedData.getIdentifier(position()).getPointer();
                auto positionOffset = loadedData.getIdentifier(totalCellIdx()).getPointer();

                // grid-strided loop over the chunked data
                for(int dataBlock = worker.blockDomIdx(); dataBlock < numDataBlocks; dataBlock += worker.gridDomSize())
                {
                    auto dataBlockOffset = dataBlock * blockDomSize;
                    auto forEach = pmacc::lockstep::makeForEach(worker);
                    forEach(
                        [&](uint32_t const inBlockIdx)
                        {
                            auto idx = dataBlockOffset + inBlockIdx;
                            if(idx < size)
                            {
                                auto& positionVec = position_[idx];
                                auto& positionOffsetVec = positionOffset[idx];
                                char filterCurrent = filterKeep;
                                for(size_t d = 0; d < simDim; ++d)
                                {
                                    auto positionInD = positionVec[d] + positionOffsetVec[d];
                                    if(positionInD < patchTotalOffset[d] || positionInD > patchUpperCorner[d])
                                    {
                                        filterCurrent = filterRemove;
                                        break;
                                    }
                                }
                                filterOut[idx] = filterCurrent;
                            }
                        });
                }
            }
        };

        /** Load species from openPMD checkpoint storage
         *
         * @tparam T_Species type of species
         */
        template<typename T_Species>
        struct LoadSpecies
        {
        public:
            using ThisSpecies = T_Species;
            using FrameType = typename ThisSpecies::FrameType;
            using ParticleDescription = typename FrameType::ParticleDescription;
            using ParticleAttributeList = typename FrameType::ValueTypeSeq;


            /* delete multiMask and localCellIdx in openPMD particle*/
            using TypesToDelete = pmacc::mp_list<multiMask, localCellIdx>;
            using ParticleCleanedAttributeList = typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type;

            /* add totalCellIdx for openPMD particle*/
            using ParticleNewAttributeList = MakeSeq_t<ParticleCleanedAttributeList, totalCellIdx>;

            using NewParticleDescription =
                typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type;

            /** Load species from openPMD checkpoint storage
             *
             * @param params thread params
             * @param restartChunkSize number of particles processed in one kernel
             * call
             */
            HINLINE void operator()(ThreadParams* params, uint32_t const currentStep, uint32_t const restartChunkSize)
            {
                std::string const speciesName = FrameType::getName();
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) load species: %1%") % speciesName;
                DataConnector& dc = Environment<>::get().DataConnector();
                GridController<simDim>& gc = Environment<simDim>::get().GridController();

                ::openPMD::Series& series = *params->readOpenPMDSeries;
                ::openPMD::Container<::openPMD::ParticleSpecies>& particles
                    = series.iterations[currentStep].open().particles;
                ::openPMD::ParticleSpecies particleSpecies = particles[speciesName];

                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                DataSpace<simDim> cellOffsetToTotalDomain
                    = subGrid.getLocalDomain().offset + subGrid.getGlobalDomain().offset;

                /* load particle without copying particle data to host */
                auto speciesTmp = dc.get<ThisSpecies>(FrameType::getName());

                // avoid deadlock between not finished pmacc tasks and mpi calls in
                // openPMD
                eventSystem::getTransactionEvent().waitForFinished();

                auto numRanks = gc.getGlobalSize();

                auto [fullMatches, partialMatches] = getPatchIdx(params, particleSpecies);

                std::shared_ptr<uint64_t> numParticlesShared
                    = particleSpecies.particlePatches["numParticles"].load<uint64_t>();
                std::shared_ptr<uint64_t> numParticlesOffsetShared
                    = particleSpecies.particlePatches["numParticlesOffset"].load<uint64_t>();
                particles.seriesFlush();
                uint64_t* patchNumParticles = numParticlesShared.get();
                uint64_t* patchNumParticlesOffset = numParticlesOffsetShared.get();

                {
                    uint64_t totalNumParticles = std::transform_reduce(
                        fullMatches.begin(),
                        fullMatches.end(),
                        0,
                        /* reduce = */ [](uint64_t left, uint64_t right) { return left + right; },
                        /* transform = */ [patchNumParticles](size_t patchIdx)
                        { return patchNumParticles[patchIdx]; });

                    log<picLog::INPUT_OUTPUT>("openPMD: malloc mapped memory: %1%") % speciesName;

                    using FrameType = Frame<OperatorCreateVectorBox, NewParticleDescription>;
                    using BufferType = Frame<OperatorCreateAlpakaBuffer, NewParticleDescription>;

                    BufferType buffers;
                    FrameType mappedFrame;

                    uint64_t maxChunkSize = std::min(static_cast<uint64_t>(restartChunkSize), totalNumParticles);

                    /*malloc mapped memory*/
                    meta::ForEach<typename NewParticleDescription::ValueTypeSeq, MallocMappedMemory<boost::mpl::_1>>
                        mallocMem;
                    mallocMem(buffers, mappedFrame, maxChunkSize);

                    for(size_t const patchIdx : fullMatches)
                    {
                        uint64_t particleOffset = patchNumParticlesOffset[patchIdx];
                        uint64_t numParticles = patchNumParticles[patchIdx];

                        log<picLog::INPUT_OUTPUT>("openPMD: Loading %1% particles from offset %2%")
                            % (long long unsigned) totalNumParticles % (long long unsigned) particleOffset;


                        uint32_t const numLoadIterations
                            = maxChunkSize == 0u ? 0u : alpaka::core::divCeil(numParticles, maxChunkSize);

                        for(uint64_t loadRound = 0u; loadRound < numLoadIterations; ++loadRound)
                        {
                            auto particleLoadOffset = particleOffset + loadRound * maxChunkSize;
                            auto numLeftParticles = numParticles - loadRound * maxChunkSize;

                            auto numParticlesCurrentBatch = std::min(numLeftParticles, maxChunkSize);

                            log<picLog::INPUT_OUTPUT>("openPMD: (begin) load species %1% round: %2%/%3%") % speciesName
                                % (loadRound + 1) % numLoadIterations;
                            if(numParticlesCurrentBatch != 0)
                            {
                                meta::ForEach<
                                    typename NewParticleDescription::ValueTypeSeq,
                                    LoadParticleAttributesFromOpenPMD<boost::mpl::_1>>
                                    loadAttributes;
                                loadAttributes(
                                    params,
                                    mappedFrame,
                                    particleSpecies,
                                    particleLoadOffset,
                                    numParticlesCurrentBatch);

                                pmacc::particles::operations::splitIntoListOfFrames(
                                    *speciesTmp,
                                    mappedFrame,
                                    numParticlesCurrentBatch,
                                    cellOffsetToTotalDomain,
                                    totalCellIdx_,
                                    *(params->cellDescription),
                                    picLog::INPUT_OUTPUT());
                            }
                            log<picLog::INPUT_OUTPUT>("openPMD: ( end ) load species %1% round: %2%/%3%") % speciesName
                                % (loadRound + 1) % numLoadIterations;
                        }
                    }
                }

                {
                    SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                    pmacc::Selection<simDim> const localDomain = subGrid.getLocalDomain();
                    pmacc::Selection<simDim> const globalDomain = subGrid.getGlobalDomain();
                    /* Offset to transform local particle offsets into total offsets for all particles within the
                     * current local domain.
                     * @attention A window can be the full simulation domain or the moving window.
                     */
                    DataSpace<simDim> localToTotalDomainOffset(localDomain.offset + globalDomain.offset);

                    /* params->localWindowToDomainOffset is in PIConGPU for a restart zero but to stay generic we take
                     * the variable into account.
                     */
                    DataSpace<simDim> const patchTotalOffset
                        = localToTotalDomainOffset + params->localWindowToDomainOffset;
                    DataSpace<simDim> const patchExtent = params->window.localDimensions.size;
                    DataSpace<simDim> const patchUpperCorner = patchTotalOffset + patchExtent;

                    uint64_t totalNumParticles = std::transform_reduce(
                        partialMatches.begin(),
                        partialMatches.end(),
                        0,
                        /* reduce = */ [](uint64_t left, uint64_t right) { return left + right; },
                        /* transform = */ [patchNumParticles](size_t patchIdx)
                        { return patchNumParticles[patchIdx]; });

                    log<picLog::INPUT_OUTPUT>("openPMD: malloc mapped memory for partial patches: %1%") % speciesName;

                    using FrameType = Frame<OperatorCreateVectorBox, NewParticleDescription>;
                    using BufferType = Frame<OperatorCreateAlpakaBuffer, NewParticleDescription>;

                    BufferType buffers;
                    FrameType mappedFrame;

                    uint64_t maxChunkSize = std::min(static_cast<uint64_t>(restartChunkSize), totalNumParticles);

                    /*malloc mapped memory*/
                    meta::ForEach<typename NewParticleDescription::ValueTypeSeq, MallocMappedMemory<boost::mpl::_1>>
                        mallocMem;
                    mallocMem(buffers, mappedFrame, maxChunkSize);
                    constexpr bool isMappedMemorySupported
                        = alpaka::hasMappedBufSupport<::alpaka::Platform<pmacc::ComputeDevice>>;
                    PMACC_VERIFY_MSG(isMappedMemorySupported, "Device must support mapped memory!");
                    // alpaka::Buf<HostDevice, char, AlpakaDim<DIM1>> filter;
                    auto filter = alpaka::allocMappedBuf<char, MemIdxType>(
                        manager::Device<HostDevice>::get().current(),
                        manager::Device<ComputeDevice>::get().getPlatform(),
                        MemSpace<DIM1>(maxChunkSize).toAlpakaMemVec());
                    auto remap = alpaka::allocMappedBuf<MemIdxType, MemIdxType>(
                        manager::Device<HostDevice>::get().current(),
                        manager::Device<ComputeDevice>::get().getPlatform(),
                        MemSpace<DIM1>(maxChunkSize).toAlpakaMemVec());
                    for(size_t const patchIdx : partialMatches)
                    {
                        uint64_t particleOffset = patchNumParticlesOffset[patchIdx];
                        uint64_t numParticles = patchNumParticles[patchIdx];

                        log<picLog::INPUT_OUTPUT>("openPMD: Loading up to %1% particles from offset %2%")
                            % (long long unsigned) totalNumParticles % (long long unsigned) particleOffset;


                        uint32_t const numLoadIterations
                            = maxChunkSize == 0u ? 0u : alpaka::core::divCeil(numParticles, maxChunkSize);

                        for(uint64_t loadRound = 0u; loadRound < numLoadIterations; ++loadRound)
                        {
                            auto particleLoadOffset = particleOffset + loadRound * maxChunkSize;
                            auto numLeftParticles = numParticles - loadRound * maxChunkSize;

                            auto numParticlesCurrentBatch = std::min(numLeftParticles, maxChunkSize);

                            log<picLog::INPUT_OUTPUT>("openPMD: (begin) load species %1% round: %2%/%3%") % speciesName
                                % (loadRound + 1) % numLoadIterations;
                            if(numParticlesCurrentBatch != 0)
                            {
                                meta::ForEach<
                                    typename NewParticleDescription::ValueTypeSeq,
                                    LoadParticleAttributesFromOpenPMD<boost::mpl::_1>>
                                    loadAttributes;
                                loadAttributes(
                                    params,
                                    mappedFrame,
                                    particleSpecies,
                                    particleLoadOffset,
                                    numParticlesCurrentBatch);

                                // now filter

                                constexpr char filterKeep{1}, filterRemove{0};
                                PMACC_LOCKSTEP_KERNEL(KernelFilterParticles{})
                                    .config<DIM1>(pmacc::math::Vector{numParticlesCurrentBatch})(
                                        mappedFrame,
                                        numParticlesCurrentBatch,
                                        patchTotalOffset,
                                        patchUpperCorner,
                                        filterKeep,
                                        filterRemove,
                                        alpaka::getPtrNative(filter));
                                eventSystem::getTransactionEvent().waitForFinished();

                                // This part is inherently sequential, keep it on CPU
                                // (maybe run also this on GPU to avoid multiple data copies)
                                // std::cout << "REMAP: ";
                                MemIdxType remapCurrent = 0;
                                for(size_t particleIndex = 0; particleIndex < numParticlesCurrentBatch;
                                    ++particleIndex)
                                {
                                    if(filter[particleIndex] == filterKeep)
                                    {
                                        remap[particleIndex] = remapCurrent++;
                                        // std::cout << '1';
                                    }
                                    else
                                    {
                                        remap[particleIndex] = std::numeric_limits<MemIdxType>::max();
                                        // std::cout << '0';
                                    }
                                }
                                // std::cout << std::endl;

                                meta::ForEach<
                                    typename NewParticleDescription::ValueTypeSeq,
                                    RedistributeFilteredParticles<boost::mpl::_1>>
                                    redistributeFilteredParticles;
                                redistributeFilteredParticles(
                                    mappedFrame,
                                    filter,
                                    remap,
                                    numParticlesCurrentBatch,
                                    filterRemove);

                                std::cout << "Filtered " << remapCurrent << " out of " << numParticlesCurrentBatch
                                          << " particles" << std::endl;

                                pmacc::particles::operations::splitIntoListOfFrames(
                                    *speciesTmp,
                                    mappedFrame,
                                    remapCurrent, // !! not numParticlesCurrentBatch, filtered vs. unfiltered number
                                    cellOffsetToTotalDomain,
                                    totalCellIdx_,
                                    *(params->cellDescription),
                                    picLog::INPUT_OUTPUT());
                            }
                            log<picLog::INPUT_OUTPUT>("openPMD: ( end ) load species %1% round: %2%/%3%") % speciesName
                                % (loadRound + 1) % numLoadIterations;
                        }
                    }
                }
                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) load species: %1%") % speciesName;
            }

        private:
            // o: offset, e: extent, u: upper corner (= o+e)
            static std::pair<DataSpace<simDim>, DataSpace<simDim>> intersect(
                DataSpace<simDim> const& o1,
                DataSpace<simDim> const& e1,
                DataSpace<simDim> const& o2,
                DataSpace<simDim> const& e2)
            {
                // Convert extents into upper coordinates
                auto u1 = o1 + e1;
                auto u2 = o2 + e2;

                DataSpace<simDim> intersect_o, intersect_u, intersect_e;
                for(unsigned d = 0; d < simDim; ++d)
                {
                    intersect_o[d] = std::max(o1[d], o2[d]);
                    intersect_u[d] = std::min(u1[d], u2[d]);
                    intersect_e[d] = intersect_u[d] > intersect_o[d] ? intersect_u[d] - intersect_o[d] : 0;
                }
                return {intersect_o, intersect_e};
            }

            /** get index for particle data within the openPMD patch data
             *
             * It is not possible to assume that we can use the MPI rank to load the particle data.
             * There is no guarantee that the MPI rank is corresponding to the position within
             * the simulation volume.
             *
             * Use patch information offset and extent to find the index which should be used
             * to load openPMD particle patch data.
             *
             * @return index of the particle patch within the openPMD data
             */
            HINLINE std::pair<std::deque<size_t>, std::deque<size_t>> getPatchIdx(
                ThreadParams* params,
                ::openPMD::ParticleSpecies particleSpecies)
            {
                std::string const name_lookup[] = {"x", "y", "z"};

                size_t patches = particleSpecies.particlePatches["numParticles"].getExtent()[0];

                std::vector<DataSpace<simDim>> offsets(patches);
                std::vector<DataSpace<simDim>> extents(patches);

                // transform openPMD particle patch data into PIConGPU data objects
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    std::shared_ptr<uint64_t> patchOffsetsInfoShared
                        = particleSpecies.particlePatches["offset"][name_lookup[d]].load<uint64_t>();
                    std::shared_ptr<uint64_t> patchExtentsInfoShared
                        = particleSpecies.particlePatches["extent"][name_lookup[d]].load<uint64_t>();
                    particleSpecies.seriesFlush();
                    for(size_t i = 0; i < patches; ++i)
                    {
                        offsets[i][d] = patchOffsetsInfoShared.get()[i];
                        extents[i][d] = patchExtentsInfoShared.get()[i];
                    }
                }

                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                pmacc::Selection<simDim> const localDomain = subGrid.getLocalDomain();
                pmacc::Selection<simDim> const globalDomain = subGrid.getGlobalDomain();
                /* Offset to transform local particle offsets into total offsets for all particles within the
                 * current local domain.
                 * @attention A window can be the full simulation domain or the moving window.
                 */
                DataSpace<simDim> localToTotalDomainOffset(localDomain.offset + globalDomain.offset);

                /* params->localWindowToDomainOffset is in PIConGPU for a restart zero but to stay generic we take
                 * the variable into account.
                 */
                DataSpace<simDim> const patchTotalOffset
                    = localToTotalDomainOffset + params->localWindowToDomainOffset;
                DataSpace<simDim> const patchExtent = params->window.localDimensions.size;
                math::Vector<bool, simDim> true_;
                for(unsigned d = 0; d < simDim; ++d)
                {
                    true_[d] = true;
                }

                // search the patch index based on the offset and extents of local domain size
                std::deque<size_t> fullMatches;
                std::deque<size_t> partialMatches;
                size_t noMatches = 0;
                for(size_t i = 0; i < patches; ++i)
                {
                    // std::cout << "Comp.: " << patchTotalOffset << " - " << (patchTotalOffset + patchExtent)
                    //           << "\tAGAINST " << offsets[i] << " - " << (offsets[i] + extents[i])
                    //           << "\toffsets: " << (patchTotalOffset <= offsets[i])
                    //           << ", extents: " << ((offsets[i] + extents[i]) <= (patchTotalOffset + patchExtent)) <<
                    //           '\n';
                    if((patchTotalOffset <= offsets[i]) == true_
                       && ((offsets[i] + extents[i]) <= (patchTotalOffset + patchExtent)) == true_)
                    {
                        fullMatches.emplace_back(i);
                    }
                    else if(
                        intersect(offsets[i], extents[i], patchTotalOffset, patchExtent).second.productOfComponents()
                        != 0)
                    {
                        partialMatches.emplace_back(i);
                    }
                    else
                    {
                        ++noMatches;
                    }
                }

                std::cout << "\n\n"
                          << fullMatches.size() << " full matches, " << partialMatches.size() << " partial matches, "
                          << noMatches << " unmatched." << std ::endl;

                return std::make_pair(std::move(fullMatches), std::move(partialMatches));
            }
        };


    } /* namespace openPMD */

} /* namespace picongpu */

#endif
