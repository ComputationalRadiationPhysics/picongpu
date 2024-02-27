/* Copyright 2019-2023 Rene Widera, Pawel Ordyna
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

#include "picongpu/simulation_defines.hpp"

#include <pmacc/random/distributions/Uniform.hpp>
#include <pmacc/verify.hpp>

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include <mallocMC/mallocMC.hpp>
#endif

namespace picongpu::particles::collision
{
    namespace detail
    {
        /** Access all particles in a cell.
         *
         * An accessor provides access to all particles within a cell (NOT supercell).
         *
         * @tparam T_FramePtrType frame pointer type
         */
        template<typename T_ParBox>
        struct ParticleAccessor
        {
            using FramePtrType = typename T_ParBox::FramePtr;
            FramePtrType m_framePtr;
            uint32_t* m_parIdxList = nullptr;
            uint32_t m_numPars = 0u;
            T_ParBox const& m_parBox;

            static constexpr uint32_t frameSize = FramePtrType::type::frameSize;

            DINLINE ParticleAccessor(
                T_ParBox const& parBox,
                uint32_t* parIdxList,
                uint32_t const numParticles,
                FramePtrType const& framePtr)
                : m_framePtr(framePtr)
                , m_parIdxList(parIdxList)
                , m_numPars(numParticles)
                , m_parBox(parBox)
            {
            }

            /** Number of particles within the cell
             */
            DINLINE uint32_t size() const
            {
                return m_numPars;
            }

            /** get particle
             *
             * @param idx index of the particle, range [0;size())
             */
            DINLINE auto operator[](uint32_t idx) const
            {
                uint32_t const inSuperCellIdx = m_parIdxList[idx];
                uint32_t const skipFrames = inSuperCellIdx / frameSize;

                auto frame = m_framePtr;
                for(uint32_t i = 0; i < skipFrames; ++i)
                    frame = m_parBox.getNextFrame(frame);
                return frame[inSuperCellIdx % frameSize];
            }
        };

        /* Storage for particle.
         *
         * Storage for particles of collision eligible macro particles. It comes with a simple
         * shuffling algorithm.
         */
        template<uint32_t T_numElem>
        struct ListEntry
        {
        private:
            // contiguous particle indices
            memory::Array<uint32_t*, T_numElem> particleList;

            //! number of particles per cell
            memory::Array<uint32_t, T_numElem> numParticles;

        public:
            DINLINE uint32_t& size(uint32_t cellIdx)
            {
                return numParticles[cellIdx];
            }

            /** Get particle index array.
             *
             * @param cellIdx index of the cell within the supercell
             */
            DINLINE uint32_t* particleIds(uint32_t cellIdx)
            {
                return particleList[cellIdx];
            }

            /* Initialize storage, allocate memory.
             *
             * @attention This is a collective method and must be called by any worker.
             *            Worker will not be synchronized.
             *
             * @param worker alpaka accelerator
             * @param deviceHeapHandle heap handle used for allocating storage on device
             * @param bp particle box
             * @param superCellIdx supercell index, relative to the guard origin
             * @param numParArray array with number of particles for each cell
             */
            template<
                typename T_Worker,
                typename T_DeviceHeapHandle,
                typename T_ParticlesBox,
                typename T_NumParticlesArray>
            DINLINE void init(
                T_Worker const& worker,
                T_DeviceHeapHandle& deviceHeapHandle,
                T_ParticlesBox pb,
                DataSpace<simDim> superCellIdx,
                T_NumParticlesArray& numParArray)
            {
                auto forEachCell = lockstep::makeForEach<T_numElem>(worker);
                // memory for particle indices
                forEachCell(
                    [&](uint32_t const cellIdx)
                    {
                        particleList[cellIdx] = nullptr;
                        // reset class counter
                        numParticles[cellIdx] = 0u;
                        uint32_t numParsInCell = numParArray[cellIdx];

                        // Chunk size in bytes based on the typical initial number particles per cell.
                        constexpr uint32_t chunkSizePerCell = cellListChunkSize * sizeof(uint32_t);
                        particleList[cellIdx] = (uint32_t*)
                            allocMem<chunkSizePerCell>(worker, sizeof(uint32_t) * numParsInCell, deviceHeapHandle);
                    });
            }

            /* Update cell particle index list
             *
             * Update the contiguous index list per cell with particle indices.
             */
            template<typename T_Worker, typename T_ForEachFrame, typename T_Filter>
            DINLINE void updateLinkedList(T_Worker const& worker, T_ForEachFrame forEachFrame, T_Filter& filter)
            {
                uint32_t frameIdx = 0u;

                forEachFrame(
                    [&](auto const& lockstepWorker, auto& frameCtx)
                    {
                        // loop over all particles in the frame
                        auto forEachParticleInFrame = forEachFrame.lockstepForEach();

                        forEachParticleInFrame(
                            [&](uint32_t const linearIdx, auto& frame)
                            {
                                auto particle = frame[linearIdx];

                                if(particle[multiMask_] != 0 && filter(worker, particle))
                                {
                                    auto parLocalIndex = particle[localCellIdx_];
                                    uint32_t parOffset = cupla::atomicAdd(
                                        worker.getAcc(),
                                        &size(parLocalIndex),
                                        1u,
                                        ::alpaka::hierarchy::Threads{});
                                    uint32_t const parInSuperCellIdx = frameIdx * numFrameSlots + linearIdx;
                                    particleIds(parLocalIndex)[parOffset] = parInSuperCellIdx;
                                }
                            },
                            frameCtx);
                        // increment frame index after all particles in the current frame are processed
                        ++frameIdx;
                    });
            }

            /* Release allocated heap memory.
             *
             * @param acc alpaka accelerator
             * @param deviceHeapHandle Heap handle used for allocating storage on device.
             */
            template<typename T_Worker, typename T_DeviceHeapHandle>
            DINLINE void finalize(T_Worker const& worker, T_DeviceHeapHandle& deviceHeapHandle)
            {
                auto forEachCell = lockstep::makeForEach<T_numElem>(worker);
                // memory for particle indices
                forEachCell(
                    [&](uint32_t const cellIdx)
                    {
                        if(particleList[cellIdx] != nullptr)
                        {
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
                            deviceHeapHandle.free(worker.getAcc(), (void*) particleList[cellIdx]);
#else
                            delete(particleList[cellIdx]);
#endif
                            particleList[cellIdx] = nullptr;
                        }
                    });
            }


            /** Creates an accessor to access particles within a cell
             *
             * @tparam T_FramePtrType
             * @param cellIdx cell index within the supercell, range [0, number of cells in supercell)
             * @return accessor to access particles via index
             */
            template<typename T_ParBox>
            DINLINE auto getParticlesAccessor(
                T_ParBox const& parBox,
                DataSpace<simDim> const& superCellIdx,
                uint32_t cellIdx)
            {
                using FramePtrType = typename T_ParBox::FramePtr;
                return ParticleAccessor<T_ParBox>(
                    parBox,
                    particleIds(cellIdx),
                    size(cellIdx),
                    parBox.getFirstFrame(superCellIdx));
            }

        private:
            /** Allocate a chunk of memory
             *
             * @tparam T_chunkBytes number of bytes, the allocated size will be a multiple of the chunk bytes
             * size
             * @param bytes number of bytes to allocate
             * @return pointer to allocated data, nullptr in case bytes is zero or there is not enough memory
             * in the heap available.
             */
            template<uint32_t T_chunkBytes, typename T_DeviceHeapHandle, typename T_Worker>
            DINLINE void* allocMem(T_Worker const& worker, uint32_t const bytes, T_DeviceHeapHandle& deviceHeapHandle)
            {
                void* ptr = nullptr;
                uint32_t const numChunks = (bytes + T_chunkBytes - 1u) / T_chunkBytes;
                uint32_t const allocBytes = numChunks * T_chunkBytes;
                if(bytes != 0u)
                {
                    const int maxTries = 13; // magic number is not performance critical
                    for(int numTries = 0; numTries < maxTries; ++numTries)
                    {
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
                        ptr = deviceHeapHandle.malloc(worker.getAcc(), allocBytes);
#else // No cuda or hip means the device heap is the host heap.
                        return (void**) new uint8_t[allocBytes];
#endif

                        if(ptr != nullptr)
                        {
                            break;
                        }
                    }
                    PMACC_DEVICE_VERIFY_MSG(
                        ptr != nullptr,
                        "Error: Out of device heap memory in %s:%u\n",
                        __FILE__,
                        __LINE__);
                }
                return ptr;
            }
        };

        //! Count number of particles per cell in the supercell
        template<typename T_Worker, typename T_ForEachParticle, typename T_Array, typename T_Filter>
        DINLINE void particlesCntHistogram(
            T_Worker const& worker,
            T_ForEachParticle forEachParticle,
            T_Array& nppc,
            T_Filter& filter)
        {
            forEachParticle(
                [&](auto const& lockstepWorker, auto& particle)
                {
                    if(filter(worker, particle))
                    {
                        auto parLocalIndex = particle[localCellIdx_];
                        cupla::atomicAdd(worker.getAcc(), &nppc[parLocalIndex], 1u, ::alpaka::hierarchy::Threads{});
                    }
                });
        }

        /* Swap two list entries.
         *
         * @param v0 index of the 1st entry to swap
         * @param v1 index of the 2nd entry to swap
         */
        DINLINE void swap(uint32_t& v0, uint32_t& v1)
        {
            uint32_t tmp = v0;
            v0 = v1;
            v1 = tmp;
        }

        /* Shuffle list entries
         *
         * Shuffle the given array with indices.
         *
         * @param worker lockstep worker
         * @param ptr pointer to a list of indices
         * @param numElements number of particles accessible via ptr
         * @param rngHandle random number generator handle
         */
        template<typename T_Worker, typename T_RngHandle>
        DINLINE void shuffle(T_Worker const& worker, uint32_t* ptr, uint32_t numElements, T_RngHandle& rngHandle)
        {
            using UniformUint32_t = pmacc::random::distributions::Uniform<uint32_t>;
            auto rng = rngHandle.template applyDistribution<UniformUint32_t>();
            // shuffle the particle lookup table
            for(uint32_t i = numElements; i > 1; --i)
            {
                /* modulo is not perfect but okish,
                 * because of the loop head mod zero is not possible
                 */
                uint32_t p = rng(worker) % i;
                if(i - 1 != p)
                    detail::swap(ptr[i - 1], ptr[p]);
            }
        }

        template<
            typename T_Worker,
            typename T_ForEachCell,
            typename T_ParticlesBox,
            typename T_DeviceHeapHandle,
            typename T_Array,
            typename T_Filter,
            uint32_t T_numListEntryCells>
        DINLINE void prepareList(
            T_Worker const& worker,
            T_ForEachCell forEachCell,
            T_ParticlesBox pb,
            DataSpace<simDim> superCellIdx,
            T_DeviceHeapHandle deviceHeapHandle,
            ListEntry<T_numListEntryCells>& listEntrys,
            T_Array& nppc,

            T_Filter filter)
        {
            // Initialize nppc with zeros.
            forEachCell([&](uint32_t const linearIdx) { nppc[linearIdx] = 0u; });
            worker.sync();
            // Count eligible
            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, pb, superCellIdx);
            particlesCntHistogram(worker, forEachParticle, nppc, filter);
            worker.sync();
            // memory for particle indices
            listEntrys.init(worker, deviceHeapHandle, pb, superCellIdx, nppc);
            worker.sync();
            auto forEachFrame
                = pmacc::particles::algorithm::acc::makeForEachFrame<pmacc::particles::algorithm::acc::Forward>(
                    worker,
                    pb,
                    superCellIdx);
            listEntrys.updateLinkedList(worker, forEachFrame, filter);
            worker.sync();
        }
    } // namespace detail
} // namespace picongpu::particles::collision
