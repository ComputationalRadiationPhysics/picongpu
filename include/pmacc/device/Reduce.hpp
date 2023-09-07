/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
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


#include "pmacc/device/reduce/Kernel.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/math/operation.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/traits/GetValueType.hpp"
#include "pmacc/types.hpp"

#include <memory>
#include <type_traits>

namespace pmacc
{
    namespace device
    {
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
            {
                reduceBuffer = std::make_unique<GridBuffer<char, DIM1>>(DataSpace<DIM1>(byte));
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
                using Type = typename std::remove_const_t<
                    typename std::remove_reference_t<typename traits::GetValueType<Src>::ValueType>>;

                uint32_t blockcount = optimalThreadsPerBlock(n, sizeof(Type));

                uint32_t n_buffer = byte / sizeof(Type);

                uint32_t threads = n_buffer * blockcount
                    * 2; /* x2 is used thus we can use all byte in Buffer, after we calculate threads/2 */


                if(threads > n)
                    threads = n;
                auto* dest = (Type*) reduceBuffer->getDeviceBuffer().getBasePointer();

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
                    pmacc::math::operation::Assign());
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
                            pmacc::math::operation::Assign());
                    }

                    n = blocks;
                    blockcount = optimalThreadsPerBlock(n, sizeof(Type));
                    blocks = n / 2 / blockcount;
                    if(blocks == 0 && n > 1)
                        blocks = 1;
                }

                reduceBuffer->deviceToHost();
                eventSystem::getTransactionEvent().waitForFinished();
                return *((Type*) (reduceBuffer->getHostBuffer().getBasePointer()));
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
            HINLINE void callReduceKernel(uint32_t blocks, uint32_t threads, uint32_t sharedMemSize, T_Args&&... args)
            {
                if(threads >= 512u)
                {
                    auto workerCfg = lockstep::makeWorkerCfg<512u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 512u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
                }
                else if(threads >= 256u)
                {
                    auto workerCfg = lockstep::makeWorkerCfg<256u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 256u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
                }
                else if(threads >= 128u)
                {
                    auto workerCfg = lockstep::makeWorkerCfg<128u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 128u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
                }
                else if(threads >= 64u)
                {
                    auto workerCfg = lockstep::makeWorkerCfg<64u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 64u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
                }
                else if(threads >= 32u)
                {
                    auto workerCfg = lockstep::makeWorkerCfg<32u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 32u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
                }
                else if(threads >= 16u)
                {
                    auto workerCfg = lockstep::makeWorkerCfg<16u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 16u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
                }
                else if(threads >= 8u)
                {
                    auto workerCfg = lockstep::makeWorkerCfg<8u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 8u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
                }
                else if(threads >= 4u)
                {
                    auto workerCfg = lockstep::makeWorkerCfg<4u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 4u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
                }
                else if(threads >= 2u)
                {
                    auto workerCfg = lockstep::makeWorkerCfg<2u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 2u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
                }
                else
                {
                    auto workerCfg = lockstep::makeWorkerCfg<1u>();
                    PMACC_LOCKSTEP_KERNEL(reduce::Kernel<Type, 1u>{}, workerCfg)
                    (blocks, sharedMemSize)(args...);
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
            std::unique_ptr<GridBuffer<char, DIM1>> reduceBuffer;
            /*buffer size limit in bytes on gpu*/
            uint32_t byte;
            /*shared memory limit in byte for one block*/
            uint32_t sharedMemByte;
        };

    } // namespace device
} // namespace pmacc
