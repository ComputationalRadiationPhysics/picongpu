/* Copyright 2022 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#    if _OPENACC < 201306
#        error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC 2.0 or higher!
#    endif

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

// Implementation details.
#    include <alpaka/acc/AccOacc.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/core/Tuple.hpp>
#    include <alpaka/ctx/block/CtxBlockOacc.hpp>
#    include <alpaka/dev/DevOacc.hpp>
#    include <alpaka/idx/MapIdx.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <algorithm>
#    include <functional>
#    include <stdexcept>
#    include <type_traits>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

namespace alpaka
{
    //! The OpenACC accelerator execution task.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelOacc final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelOacc(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(kernelFnObj)
            , m_args(std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        //! Executes the kernel function object.
        ALPAKA_FN_HOST auto operator()(DevOacc const& dev) const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(*this);
            auto const blockThreadExtent = getWorkDiv<Block, Threads>(*this);
            auto const threadElemExtent = getWorkDiv<Thread, Elems>(*this);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            std::cout << "m_gridBlockExtent=" << this->m_gridBlockExtent << "\tgridBlockExtent=" << gridBlockExtent
                      << std::endl;
            std::cout << "m_blockThreadExtent=" << this->m_blockThreadExtent
                      << "\tblockThreadExtent=" << blockThreadExtent << std::endl;
            std::cout << "m_threadElemExtent=" << this->m_threadElemExtent << "\tthreadElemExtent=" << threadElemExtent
                      << std::endl;
#    endif

            // Get the size of the block shared dynamic memory.
            auto const blockSharedMemDynSizeBytes = core::apply(
                [&](ALPAKA_DECAY_T(TArgs) const&... args)
                {
                    return getBlockSharedMemDynSizeBytes<AccOacc<TDim, TIdx>>(
                        m_kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...);
                },
                m_args);

#    if ALPAKA_DEBUG > ALPAKA_DEBUG_MINIMAL
            std::cout << __func__ << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                      << std::endl;
#    endif
            // The number of blocks in the grid.
            TIdx const gridBlockCount(gridBlockExtent.prod());
            // The number of threads in a block.
            TIdx const blockThreadCount(blockThreadExtent.prod());

            if(gridBlockCount == 0 || blockThreadCount == 0)
            { //! empty grid is a NOP
                return;
            }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            std::cout << "threadElemCount=" << threadElemExtent[0u] << "\tgridBlockCount=" << gridBlockCount
                      << std::endl;
#    endif
            auto argsD = m_args;
            auto kernelFnObj = m_kernelFnObj;
            dev.makeCurrent();

            std::uint32_t blocksLock[2] = {0u, 0u};
            std::uint32_t* gridsLock = dev.gridsLock();

#    pragma acc parallel num_workers(blockThreadCount) copyin(                                                        \
        threadElemExtent,                                                                                             \
        blockThreadExtent,                                                                                            \
        gridBlockExtent,                                                                                              \
        argsD,                                                                                                        \
        blockSharedMemDynSizeBytes,                                                                                   \
        blocksLock [0:2],                                                                                             \
        kernelFnObj) default(present) deviceptr(gridsLock)
            {
                {
#    pragma acc loop gang
                    for(TIdx b = 0u; b < gridBlockCount; ++b)
                    {
                        CtxBlockOacc<TDim, TIdx> blockShared(
                            gridBlockExtent,
                            blockThreadExtent,
                            threadElemExtent,
                            b,
                            blockSharedMemDynSizeBytes,
                            gridsLock,
                            blocksLock);

// Execute the threads in parallel.

// Parallel execution of the threads in a block is required because when
// syncBlockThreads is called all of them have to be done with their work up
// to this line.  So we have to spawn one OS thread per thread in a block.
//! \warning The OpenACC is technically allowed to ignore the value in the num_workers clause
//! and could run fewer threads. The standard provides no way to check how many worker threads are running.
//! If fewer threads are run, syncBlockThreads will dead-lock. It is up to the developer/user
//! to choose a blockThreadCount which the runtime will respect.
#    pragma acc loop worker
                        for(TIdx w = 0; w < blockThreadCount; ++w)
                        {
                            AccOacc<TDim, TIdx> acc(w, blockShared);

                            core::apply(
                                [kernelFnObj, &acc](typename std::decay<TArgs>::type const&... args)
                                { kernelFnObj(acc, args...); },
                                argsD);
                        }
                        freeSharedVars(blockShared);
                    }
                }
            }
        }

    private:
        TKernelFnObj m_kernelFnObj;
        core::Tuple<std::decay_t<TArgs>...> m_args;
    };

    namespace trait
    {
        //! The OpenACC execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccOacc<TDim, TIdx>;
        };

        //! The OpenACC execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevOacc;
        };

        //! The OpenACC execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //! The OpenACC execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfOacc;
        };

        //! The OpenACC execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };

        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<QueueOaccBlocking, TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueOaccBlocking& queue,
                TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);

                queue.m_spQueueImpl->m_bCurrentlyExecutingTask = true;

                task(queue.m_spQueueImpl->m_dev);

                queue.m_spQueueImpl->m_bCurrentlyExecutingTask = false;
            }
        };

        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<QueueOaccNonBlocking, TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueOaccNonBlocking& queue,
                TaskKernelOacc<TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                queue.m_spQueueImpl->m_workerThread->enqueueTask([&queue, task]()
                                                                 { task(queue.m_spQueueImpl->m_dev); });
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
