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


#include "pmacc/lockstep.hpp"
#include "pmacc/memory/buffers/GridBuffer.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/traits/GetValueType.hpp"
#include "pmacc/types.hpp"

#include <pmacc/math/operation.hpp>


namespace pmacc
{
    namespace device
    {
        namespace reduce
        {
            /** kernel to reduce elements within a buffer
             *
             * @tparam type element type within the buffer
             * @tparam T_blockSize minimum number of elements which will be reduced
             *                     within a CUDA block
             */
            template<typename Type, uint32_t T_blockSize>
            struct Kernel
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
                 * @tparam T_Worker lockstep worker type
                 *
                 * @param worker lockstep worker
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
                    typename T_Worker>
                DINLINE void operator()(
                    T_Worker const& worker,
                    T_SrcBuffer const& srcBuffer,
                    uint32_t const bufferSize,
                    T_DestBuffer destBuffer,
                    T_Functor func,
                    T_DestFunctor destFunc) const
                {
                    uint32_t const numGlobalVirtualThreadCount
                        = pmacc::device::getGridSize(worker.getAcc()).x() * T_blockSize;

                    Type* s_mem = ::alpaka::getDynSharedMem<Type>(worker.getAcc());

                    this->operator()(
                        worker,
                        numGlobalVirtualThreadCount,
                        srcBuffer,
                        bufferSize,
                        func,
                        s_mem,
                        device::getBlockIdx(worker.getAcc()).x());

                    lockstep::makeMaster(worker)(
                        [&]() { destFunc(worker, destBuffer[device::getBlockIdx(worker.getAcc()).x()], s_mem[0]); });
                }

                /** reduce a buffer
                 *
                 * This method can be used to reduce a chunk of an array.
                 * This method is a **collective** method and needs to be called by all
                 * threads within a alpaka block.
                 *
                 * @tparam T_SrcBuffer type of the buffer
                 * @tparam T_Functor type of the binary functor to reduce two elements
                 * @tparam T_SharedBuffer type of the shared memory buffer
                 * @tparam T_Worker lockstep worker type
                 *
                 * @param worker lockstep worker
                 * @param numReduceThreads Number of threads which working together to reduce the array.
                 *                         For a reduction within a block the value must be equal to T_blockSize
                 * @param srcBuffer a class or a pointer with the `operator[](size_t)` (one dimensional access)
                 * @param bufferSize number of elements in @p srcBuffer
                 * @param func binary functor for reduce which takes two arguments,
                 *        first argument is the source and get the new reduced value.
                 * @param sharedMem shared memory buffer with storage for `linearThreadIdxInBlock` elements,
                 *        buffer must implement `operator[](size_t)` (one dimensional access)
                 * @param blockIndex index of the alpaka block,
                 *                   for a global reduce: `device::getBlockIdx(worker.getAcc()).x()`,
                 *                   for a reduce within a block: `0`
                 *
                 * @result void the result is stored in the first slot of @p sharedMem
                 */
                template<typename T_SrcBuffer, typename T_Functor, typename T_SharedBuffer, typename T_Worker>
                DINLINE void operator()(
                    T_Worker const& worker,
                    size_t const numReduceThreads,
                    T_SrcBuffer const& srcBuffer,
                    size_t const bufferSize,
                    T_Functor const& func,
                    T_SharedBuffer& sharedMem,
                    size_t const blockIndex = 0u) const
                {
                    auto forEachBlockElem = lockstep::makeForEach<T_blockSize>(worker);

                    auto linearReduceThreadIdxCtx = forEachBlockElem(
                        [&](uint32_t const linearIdx) -> uint32_t { return blockIndex * T_blockSize + linearIdx; });

                    auto isActiveCtx = forEachBlockElem(
                        [&](auto linearReduceThreadIdx) -> bool { return linearReduceThreadIdx < bufferSize; },
                        linearReduceThreadIdxCtx);

                    forEachBlockElem(
                        [&](uint32_t const idx, bool isActive, uint32_t linearReduceThreadIdx)
                        {
                            if(isActive)
                            {
                                /*fill shared mem*/
                                Type r_value = srcBuffer[linearReduceThreadIdx];
                                /*reduce not read global memory to shared*/
                                uint32_t i = linearReduceThreadIdx + numReduceThreads;
                                while(i < bufferSize)
                                {
                                    func(worker, r_value, srcBuffer[i]);
                                    i += numReduceThreads;
                                }
                                sharedMem[idx] = r_value;
                            }
                        },
                        isActiveCtx,
                        linearReduceThreadIdxCtx);

                    worker.sync();
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

                        forEachBlockElem(
                            [&](uint32_t const linearIdx, bool& isActive, uint32_t linearReduceThreadIdx)
                            {
                                isActive = (linearReduceThreadIdx < bufferSize)
                                    && !(linearIdx != 0u && linearIdx >= active_threads);
                                if(isActive)
                                    func(worker, sharedMem[linearIdx], sharedMem[linearIdx + chunk_count]);
                            },
                            isActiveCtx,
                            linearReduceThreadIdxCtx);
                        worker.sync();
                    }
                }
            };

        } // namespace reduce
    } // namespace device
} // namespace pmacc
