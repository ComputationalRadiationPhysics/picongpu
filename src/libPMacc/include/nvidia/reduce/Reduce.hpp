/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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
 

#ifndef REDUCE_HPP
#define	REDUCE_HPP

#include "types.h"

#include "nvidia/functors/Assign.hpp"
#include "traits/GetValueType.hpp"

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

                    const uint32_t l_tid = threadIdx.x;
                    const uint32_t tid = blockIdx.x * blockDim.x + l_tid;
                    const uint32_t globalThreadCount = gridDim.x * blockDim.x;
                    extern __shared__ Type s_mem[];

                    if (tid >= src_count) return; /*end not needed threads*/
                    
                    __syncthreads(); /*wait that all shared memory is initialised*/

                    /*fill shared mem*/
                    Type r_value = src[tid];
                    //    printf("tid=%i\n", tid);
                    /*reduce not readed global memory to shared*/
                    uint32_t i = tid + globalThreadCount;
                    while (i < src_count)
                    {
                        func(r_value, src[i]);
                        i += globalThreadCount;
                    }
                    s_mem[l_tid] = r_value;
                    __syncthreads();
                    /*now reduce shared memory*/
                    uint32_t chunk_count = blockDim.x;
                    uint32_t active_threads;

                    while (chunk_count != 1)
                    {
                        const float half_threads = (float) chunk_count / 2.0f;
                        active_threads = float2uint(half_threads);
                        if (threadIdx.x != 0 && l_tid >= active_threads) return; /*end not needed threads*/


                        chunk_count = ceilf(half_threads);
                        // printf("chunk_count=%i lid=%i\n", chunk_count, l_tid);
                        //float3 s1=s_mem[l_tid];
                        // float3 s2=s_mem[l_tid + chunk_count];
                        func(s_mem[l_tid], s_mem[l_tid + chunk_count]);

                        //if (src_count == 16) printf("smem %f %i\n", s1.x(), l_tid);
                        // if (src_count == 16) printf("smem2 %f %i\n", s2.x(), l_tid + chunk_count);
                        __syncthreads();
                    }

                    func2(dest[blockIdx.x], s_mem[0]);
                    // printf("erg=%f\n", dest[blockIdx.x].x());
                }
            }

            class Reduce
            {
            public:

                /* Constructor
                 * Don't create a instance befor you have set you cuda device!
                 * @param byte how many bytes in global gpu memory can reservt for the reduce algorithm
                 * @param sharedMemByte limit the usage of shared memory per block on gpu
                 */
                HINLINE Reduce(const uint32_t byte, const uint32_t sharedMemByte = 4 * 1024) :
                byte(byte), sharedMemByte(sharedMemByte)
                {

                    reduceBuffer = new GridBuffer<char, DIM1 > (DataSpace<DIM1 > (byte));
                }

                /* Reduce elements in global gpu memeory
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
                    typedef typename traits::GetValueType<Src>::ValueType Type;

                    uint32_t blockcount = optimalThreadsPerBlock(n, sizeof (Type));

                    uint32_t n_buffer = byte / sizeof (Type);

                    uint32_t threads = n_buffer * blockcount * 2; /* x2 is used thus we can use all byte in Buffer, after we calcudlate threads/2 */



                    if (threads > n) threads = n;
                    Type* dest = (Type*) reduceBuffer->getDeviceBuffer().getBasePointer();

                    uint32_t blocks = threads / 2 / blockcount;
                    if (blocks == 0) blocks = 1;
                    //std::cout << "count=" << blockcount << " blocks=" << blocks << " n=" << n << std::endl;
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
                            // std::cout << "count=" << blockcount << " blocks=" << useBlocks << " n=" << problemSize + (blockOffset * blockcount) << std::endl;
                            __cudaKernel((kernel::reduce < Type >))(useBlocks, blockcount, blockcount * sizeof (Type))(srcPtr, problemSize, dest, func, func);
                            blocks = blockOffset*blockcount;
                        }
                        else
                        {
                            //  std::cout << "ass count=" << blockcount << " blocks=" << blocks << " n=" << n << std::endl;
                            __cudaKernel((kernel::reduce < Type >))(blocks, blockcount, blockcount * sizeof (Type))(dest, n, dest, func,
                                                                                                                    PMacc::nvidia::functors::Assign());
                        }

                        n = blocks;
                        blockcount = optimalThreadsPerBlock(n, sizeof (Type));
                        blocks = n / 2 / blockcount;
                        if (blocks == 0 && n > 1) blocks = 1;
                    }

                    // std::cout << "-------------" << blocks << std::endl;
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

                /*calculate optimal number of thredas per block with respect to shared memory limitations
                 * @param n number of elements to reduce
                 * @param sizePerElement size in bytes per elements
                 * @return optimal count of threads per block to solve the problem
                 */
                HINLINE uint32_t optimalThreadsPerBlock(uint32_t n, uint32_t sizePerElement)
                {
                    uint32_t sharedBorder = sharedMemByte / sizePerElement;
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

#endif	/* REDUCE_HPP */

