/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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


#include "nvidia/functors/Assign.hpp"
#include "traits/GetValueType.hpp"
#include "pmacc_types.hpp"
#include "memory/buffers/GridBuffer.hpp"

#include <boost/type_traits.hpp>

namespace PMacc
{
    namespace nvidia
    {
        namespace reduce
        {

            namespace kernel
            {

                template<typename Type, typename Src, typename Dest, class Functor, class Functor2>
                __global__ void reduce(
                                       Src src, const uint32_t src_count,
                                       Dest dest,
                                       Functor func, Functor2 func2)
                {
                    const uint32_t localId = threadIdx.x;
                    const uint32_t tid = blockIdx.x * blockDim.x + localId;
                    const uint32_t globalThreadCount = gridDim.x * blockDim.x;

                    /* cuda can not handle extern shared memory were the type is
                     * defined by a template
                     * - therefore we use type int for the definition (dirty but OK) */
                    extern __shared__ int s_mem_extern[];
                    /* create a pointer with the right type*/
                    Type* s_mem=(Type*)s_mem_extern;

                    if (tid >= src_count)
                        return; /*end not needed threads*/

                    /*fill shared mem*/
                    Type r_value = src[tid];
                    /*reduce not read global memory to shared*/
                    uint32_t i = tid + globalThreadCount;
                    while (i < src_count)
                    {
                        func(r_value, src[i]);
                        i += globalThreadCount;
                    }
                    s_mem[localId] = r_value;
                    __syncthreads();
                    /*now reduce shared memory*/
                    uint32_t chunk_count = blockDim.x;

                    while (chunk_count != 1)
                    {
                        /* Half number of chunks (rounded down) */
                        uint32_t active_threads = chunk_count / 2;
                        if (localId >= active_threads)
                            return; /*end not needed threads*/

                        /* New chunks is half number of chunks rounded up for uneven counts
                         * --> local_tid=0 will reduce the single element for an odd number of values at the end */
                        chunk_count = (chunk_count + 1) / 2;
                        func(s_mem[localId], s_mem[localId + chunk_count]);

                        __syncthreads();
                    }

                    func2(dest[blockIdx.x], s_mem[0]);
                }
            }

            class Reduce
            {
            public:

                /* Constructor
                 * Don't create a instance before you have set you cuda device!
                 * @param byte how many bytes in global gpu memory can reserved for the reduce algorithm
                 * @param sharedMemByte limit the usage of shared memory per block on gpu
                 */
                HINLINE Reduce(const uint32_t byte, const uint32_t sharedMemByte = 4 * 1024) :
                byte(byte), sharedMemByte(sharedMemByte), reduceBuffer(NULL)
                {

                    reduceBuffer = new GridBuffer<char, DIM1 > (DataSpace<DIM1 > (byte));
                }

                /* Reduce elements in global gpu memory
                 *
                 * @param func binary functor for reduce which takes two arguments, first argument is the source and get the new reduced value.
                 * Functor must specialize the function getMPI_Op.
                 * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one dimension access)
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
                               typename boost::remove_reference<
                                   typename traits::GetValueType<Src>::ValueType
                               >::type
                           >::type Type;

                    uint32_t blockcount = optimalThreadsPerBlock(n, sizeof (Type));

                    uint32_t n_buffer = byte / sizeof (Type);

                    uint32_t threads = n_buffer * blockcount * 2; /* x2 is used thus we can use all byte in Buffer, after we calculate threads/2 */



                    if (threads > n) threads = n;
                    Type* dest = (Type*) reduceBuffer->getDeviceBuffer().getBasePointer();

                    uint32_t blocks = threads / 2 / blockcount;
                    if (blocks == 0) blocks = 1;
                    __cudaKernel((kernel::reduce < Type >))(blocks, blockcount, blockcount * sizeof (Type))(src, n, dest, func,
                                                                                                            PMacc::nvidia::functors::Assign());
                    n = blocks;
                    blockcount = optimalThreadsPerBlock(n, sizeof (Type));
                    blocks = n / 2 / blockcount;
                    if (blocks == 0 && n > 1) blocks = 1;


                    while (blocks != 0)
                    {
                        if (blocks > 1)
                        {
                            uint32_t blockOffset = ceil((double) blocks / blockcount);
                            uint32_t useBlocks = blocks - blockOffset;
                            uint32_t problemSize = n - (blockOffset * blockcount);
                            Type* srcPtr = dest + (blockOffset * blockcount);

                            __cudaKernel((kernel::reduce < Type >))(useBlocks, blockcount, blockcount * sizeof (Type))(srcPtr, problemSize, dest, func, func);
                            blocks = blockOffset*blockcount;
                        }
                        else
                        {

                            __cudaKernel((kernel::reduce < Type >))(blocks, blockcount, blockcount * sizeof (Type))(dest, n, dest, func,
                                                                                                                    PMacc::nvidia::functors::Assign());
                        }

                        n = blocks;
                        blockcount = optimalThreadsPerBlock(n, sizeof (Type));
                        blocks = n / 2 / blockcount;
                        if (blocks == 0 && n > 1) blocks = 1;
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
                    if (threads >= 512) return 512;
                    if (threads >= 256) return 256;
                    if (threads >= 128) return 128;
                    if (threads >= 64) return 64;
                    if (threads >= 32) return 32;
                    if (threads >= 16) return 16;
                    if (threads >= 8) return 8;
                    if (threads >= 4) return 4;
                    if (threads >= 2) return 2;

                    return 1;
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
                GridBuffer<char, DIM1 > *reduceBuffer;
                /*buffer size limit in bytes on gpu*/
                uint32_t byte;
                /*shared memory limit in byte for one block*/
                uint32_t sharedMemByte;

            };
        }
    }
}
