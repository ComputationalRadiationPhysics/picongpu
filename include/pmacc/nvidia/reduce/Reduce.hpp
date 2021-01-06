/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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


#include "pmacc/nvidia/functors/Assign.hpp"
#include "pmacc/traits/GetValueType.hpp"
#include "pmacc/types.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/memory/CtxArray.hpp"
#include "pmacc/mappings/threads/ForEachIdx.hpp"
#include "pmacc/mappings/threads/IdxConfig.hpp"
#include "pmacc/mappings/threads/WorkerCfg.hpp"

#include <boost/type_traits.hpp>

namespace pmacc
{
    namespace nvidia
    {
        namespace reduce
        {
            namespace kernel
            {
                /** reduce elements within a buffer
                 *
                 * @tparam type element type within the buffer
                 * @tparam T_blockSize minimum number of elements which will be reduced
                 *                     within a CUDA block
                 * @tparam T_numWorkers number of workers
                 */
                template<typename Type, uint32_t T_blockSize, uint32_t T_numWorkers>
                struct Reduce
                {
                    /** reduce buffer
                     *
                     * This method can be used to reduce a chunk of an array.
                     * This method is a **collective** method and needs to be called by all
                     * threads within a CUDA block.
                     *
                     * @tparam T_SrcBuffer type of the buffer
                     * @tparam T_DestBuffer type of result buffer
                     * @tparam T_Functor type of the binary functor to reduce two elements to the intermediate buffer
                     * @tparam T_DestFunctor type of the binary functor to reduce two elements to @destBuffer
                     * @tparam T_Acc alpaka accelerator type
                     *
                     * @param acc alpaka accelerator
                     * @param srcBuffer a class or a pointer with the `operator[](size_t)` (one dimensional access)
                     * @param bufferSize number of elements in @p srcBuffer
                     * @param destBuffer a class or a pointer with the `operator[](size_t)` (one dimensional access),
                     *        number of elements within the buffer must be at least one
                     * @param func binary functor for reduce which takes two arguments,
                     *        first argument is the source and get the new reduced value.
                     * @param destFunc binary functor for reduce which takes two arguments,
                     *        first argument is the source and get the new reduced value.
                     *
                     * @result void intermediate results are stored in @destBuffer,
                     *         the final result is stored in the first slot of @destBuffer
                     *         if the operator is called with one CUDA block
                     */
                    template<
                        typename T_SrcBuffer,
                        typename T_DestBuffer,
                        typename T_Functor,
                        typename T_DestFunctor,
                        typename T_Acc>
                    DINLINE void operator()(
                        T_Acc const& acc,
                        T_SrcBuffer const& srcBuffer,
                        uint32_t const bufferSize,
                        T_DestBuffer destBuffer,
                        T_Functor func,
                        T_DestFunctor destFunc) const
                    {
                        using namespace mappings::threads;

                        constexpr uint32_t numWorkers = T_numWorkers;
                        uint32_t const workerIdx = cupla::threadIdx(acc).x;

                        uint32_t const numGlobalVirtualThreadCount = cupla::gridDim(acc).x * T_blockSize;
                        WorkerCfg<numWorkers> workerCfg(workerIdx);

                        sharedMemExtern(s_mem, Type);

                        this->operator()(
                            acc,
                            workerCfg,
                            numGlobalVirtualThreadCount,
                            srcBuffer,
                            bufferSize,
                            func,
                            s_mem,
                            cupla::blockIdx(acc).x);

                        using MasterOnly = IdxConfig<1, numWorkers>;

                        ForEachIdx<MasterOnly>{workerIdx}([&](uint32_t const, uint32_t const) {
                            destFunc(acc, destBuffer[cupla::blockIdx(acc).x], s_mem[0]);
                        });
                    }

                    /** reduce a buffer
                     *
                     * This method can be used to reduce a chunk of an array.
                     * This method is a **collective** method and needs to be called by all
                     * threads within a cupla block.
                     *
                     * @tparam T_SrcBuffer type of the buffer
                     * @tparam T_Functor type of the binary functor to reduce two elements
                     * @tparam T_SharedBuffer type of the shared memory buffer
                     * @tparam T_WorkerCfg worker configuration type
                     * @tparam T_Acc alpaka accelerator type
                     *
                     * @param acc alpaka accelerator
                     * @param workerCfg lockstep worker configuration
                     * @param numReduceThreads Number of threads which working together to reduce the array.
                     *                         For a reduction within a block the value must be equal to T_blockSize
                     * @param srcBuffer a class or a pointer with the `operator[](size_t)` (one dimensional access)
                     * @param bufferSize number of elements in @p srcBuffer
                     * @param func binary functor for reduce which takes two arguments,
                     *        first argument is the source and get the new reduced value.
                     * @param sharedMem shared memory buffer with storage for `linearThreadIdxInBlock` elements,
                     *        buffer must implement `operator[](size_t)` (one dimensional access)
                     * @param blockIndex index of the cupla block,
                     *                   for a global reduce: `cupla::blockIdx(acc).x`,
                     *                   for a reduce within a block: `0`
                     *
                     * @result void the result is stored in the first slot of @p sharedMem
                     */
                    template<
                        typename T_SrcBuffer,
                        typename T_Functor,
                        typename T_SharedBuffer,
                        typename T_WorkerCfg,
                        typename T_Acc>
                    DINLINE void operator()(
                        T_Acc const& acc,
                        T_WorkerCfg const workerCfg,
                        size_t const numReduceThreads,
                        T_SrcBuffer const& srcBuffer,
                        size_t const bufferSize,
                        T_Functor const& func,
                        T_SharedBuffer& sharedMem,
                        size_t const blockIndex = 0u) const
                    {
                        using namespace mappings::threads;

                        using VirtualWorkerCfg = IdxConfig<T_blockSize, T_WorkerCfg::numWorkers>;

                        pmacc::memory::CtxArray<uint32_t, VirtualWorkerCfg> linearReduceThreadIdxCtx(
                            workerCfg.getWorkerIdx(),
                            [&](uint32_t const linearIdx, uint32_t const) {
                                return blockIndex * T_blockSize + linearIdx;
                            });

                        pmacc::memory::CtxArray<bool, VirtualWorkerCfg> isActiveCtx(
                            workerCfg.getWorkerIdx(),
                            [&](uint32_t const, uint32_t const idx) {
                                return linearReduceThreadIdxCtx[idx] < bufferSize;
                            });

                        ForEachIdx<VirtualWorkerCfg> forEachVirtualThread(workerCfg.getWorkerIdx());

                        forEachVirtualThread([&](uint32_t const linearIdx, uint32_t const idx) {
                            if(isActiveCtx[idx])
                            {
                                /*fill shared mem*/
                                Type r_value = srcBuffer[linearReduceThreadIdxCtx[idx]];
                                /*reduce not read global memory to shared*/
                                uint32_t i = linearReduceThreadIdxCtx[idx] + numReduceThreads;
                                while(i < bufferSize)
                                {
                                    func(acc, r_value, srcBuffer[i]);
                                    i += numReduceThreads;
                                }
                                sharedMem[linearIdx] = r_value;
                            }
                        });

                        cupla::__syncthreads(acc);
                        /*now reduce shared memory*/
                        uint32_t chunk_count = T_blockSize;

                        while(chunk_count != 1u)
                        {
                            /* Half number of chunks (rounded down) */
                            uint32_t active_threads = chunk_count / 2u;

                            /* New chunks is half number of chunks rounded up for uneven counts
                             * --> linearThreadIdxInBlock == 0 will reduce the single element for
                             * an odd number of values at the end
                             */
                            chunk_count = (chunk_count + 1u) / 2u;

                            forEachVirtualThread([&](uint32_t const linearIdx, uint32_t const idx) {
                                isActiveCtx[idx] = (linearReduceThreadIdxCtx[idx] < bufferSize)
                                    && !(linearIdx != 0u && linearIdx >= active_threads);
                                if(isActiveCtx[idx])
                                    func(acc, sharedMem[linearIdx], sharedMem[linearIdx + chunk_count]);

                                cupla::__syncthreads(acc);
                            });
                        }
                    }
                };
            } // namespace kernel

            class Reduce
            {
            public:
                /* Constructor
                 * Don't create a instance before you have set you cupla device!
                 * @param byte how many bytes in global gpu memory can reserved for the reduce algorithm
                 * @param sharedMemByte limit the usage of shared memory per block on gpu
                 */
                HINLINE Reduce(const uint32_t byte, const uint32_t sharedMemByte = 4 * 1024)
                    : byte(byte)
                    , sharedMemByte(sharedMemByte)
                    , reduceBuffer(nullptr)
                {
                    reduceBuffer = new GridBuffer<char, DIM1>(DataSpace<DIM1>(byte));
                }

                /* Reduce elements in global gpu memory
                 *
                 * @param func binary functor for reduce which takes two arguments, first argument is the source and
                 * get the new reduced value. Functor must specialize the function getMPI_Op.
                 * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one
                 * dimensional access)
                 * @param n number of elements to reduce
                 *
                 * @return reduced value
                 */
                template<class Functor, typename Src>
                HINLINE typename traits::GetValueType<Src>::ValueType operator()(Functor func, Src src, uint32_t n)
                {
                    /* - the result of a functor can be a reference or a const value
                     * - it is not allowed to create const or reference memory
                     *   thus we remove `references` and `const` qualifiers */
                    typedef typename boost::remove_const<
                        typename boost::remove_reference<typename traits::GetValueType<Src>::ValueType>::type>::type
                        Type;

                    uint32_t blockcount = optimalThreadsPerBlock(n, sizeof(Type));

                    uint32_t n_buffer = byte / sizeof(Type);

                    uint32_t threads = n_buffer * blockcount
                        * 2; /* x2 is used thus we can use all byte in Buffer, after we calculate threads/2 */


                    if(threads > n)
                        threads = n;
                    Type* dest = (Type*) reduceBuffer->getDeviceBuffer().getBasePointer();

                    uint32_t blocks = threads / 2 / blockcount;
                    if(blocks == 0)
                        blocks = 1;
                    callReduceKernel<Type>(
                        blocks,
                        blockcount,
                        blockcount * sizeof(Type),
                        src,
                        n,
                        dest,
                        func,
                        pmacc::nvidia::functors::Assign());
                    n = blocks;
                    blockcount = optimalThreadsPerBlock(n, sizeof(Type));
                    blocks = n / 2 / blockcount;
                    if(blocks == 0 && n > 1)
                        blocks = 1;


                    while(blocks != 0)
                    {
                        if(blocks > 1)
                        {
                            uint32_t blockOffset = ceil((double) blocks / blockcount);
                            uint32_t useBlocks = blocks - blockOffset;
                            uint32_t problemSize = n - (blockOffset * blockcount);
                            Type* srcPtr = dest + (blockOffset * blockcount);

                            callReduceKernel<Type>(
                                useBlocks,
                                blockcount,
                                blockcount * sizeof(Type),
                                srcPtr,
                                problemSize,
                                dest,
                                func,
                                func);
                            blocks = blockOffset * blockcount;
                        }
                        else
                        {
                            callReduceKernel<Type>(
                                blocks,
                                blockcount,
                                blockcount * sizeof(Type),
                                dest,
                                n,
                                dest,
                                func,
                                pmacc::nvidia::functors::Assign());
                        }

                        n = blocks;
                        blockcount = optimalThreadsPerBlock(n, sizeof(Type));
                        blocks = n / 2 / blockcount;
                        if(blocks == 0 && n > 1)
                            blocks = 1;
                    }

                    reduceBuffer->deviceToHost();
                    __getTransactionEvent().waitForFinished();
                    return *((Type*) (reduceBuffer->getHostBuffer().getBasePointer()));
                }

                virtual ~Reduce()
                {
                    __delete(reduceBuffer);
                }

            private:
                /* calculate number of threads per block
                 * @param threads maximal number of threads per block
                 * @return number of threads per block
                 */
                HINLINE uint32_t getThreadsPerBlock(uint32_t threads)
                {
                    /// \todo this list is not complete
                    ///        extend it and maybe check for sm_version
                    ///        and add possible threads accordingly.
                    ///        maybe this function should be exported
                    ///        to a more general nvidia class, too.
                    if(threads >= 512)
                        return 512;
                    if(threads >= 256)
                        return 256;
                    if(threads >= 128)
                        return 128;
                    if(threads >= 64)
                        return 64;
                    if(threads >= 32)
                        return 32;
                    if(threads >= 16)
                        return 16;
                    if(threads >= 8)
                        return 8;
                    if(threads >= 4)
                        return 4;
                    if(threads >= 2)
                        return 2;

                    return 1;
                }


                /* start the reduce kernel
                 *
                 * The minimal number of elements reduced within a CUDA block is chosen at
                 * compile time.
                 */
                template<typename Type, typename... T_Args>
                HINLINE void callReduceKernel(
                    uint32_t blocks,
                    uint32_t threads,
                    uint32_t sharedMemSize,
                    T_Args&&... args)
                {
                    if(threads >= 512u)
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<512u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 512u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                    else if(threads >= 256u)
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<256u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 256u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                    else if(threads >= 128u)
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<128u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 128u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                    else if(threads >= 64u)
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<64u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 64u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                    else if(threads >= 32u)
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<32u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 32u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                    else if(threads >= 16u)
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<16u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 16u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                    else if(threads >= 8u)
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<8u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 8u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                    else if(threads >= 4u)
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<4u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 4u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                    else if(threads >= 2u)
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<2u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 2u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                    else
                    {
                        constexpr uint32_t numWorkers = traits::GetNumWorkers<1u>::value;
                        PMACC_KERNEL(kernel::Reduce<Type, 1u, numWorkers>{})
                        (blocks, numWorkers, sharedMemSize)(args...);
                    }
                }


                /*calculate optimal number of threads per block with respect to shared memory limitations
                 * @param n number of elements to reduce
                 * @param sizePerElement size in bytes per elements
                 * @return optimal count of threads per block to solve the problem
                 */
                HINLINE uint32_t optimalThreadsPerBlock(uint32_t n, uint32_t sizePerElement)
                {
                    uint32_t const sharedBorder = sharedMemByte / sizePerElement;
                    return getThreadsPerBlock(std::min(sharedBorder, n));
                }

                /*global gpu buffer for reduce steps*/
                GridBuffer<char, DIM1>* reduceBuffer;
                /*buffer size limit in bytes on gpu*/
                uint32_t byte;
                /*shared memory limit in byte for one block*/
                uint32_t sharedMemByte;
            };
        } // namespace reduce
    } // namespace nvidia
} // namespace pmacc
