/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuThreads.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>
#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/ApplyTuple.hpp>

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>
#include <tuple>
#include <type_traits>
#include <future>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace kernel
    {
        //#############################################################################
        //! The CPU threads execution task.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelCpuThreads final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        private:
            //#############################################################################
            //! The type given to the ConcurrentExecPool for yielding the current thread.
            struct ThreadPoolYield
            {
                //-----------------------------------------------------------------------------
                //! Yields the current thread.
                ALPAKA_FN_HOST static auto yield()
                -> void
                {
                    std::this_thread::yield();
                }
            };
            //#############################################################################
            // When using the thread pool the threads are yielding because this is faster.
            // Using condition variables and going to sleep is very costly for real threads.
            // Especially when the time to wait is really short (syncBlockThreads) yielding is much faster.
            using ThreadPool = alpaka::core::detail::ConcurrentExecPool<
                TIdx,
                std::thread,        // The concurrent execution type.
                std::promise,       // The promise type.
                ThreadPoolYield>;   // The type yielding the current concurrent execution.

        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelCpuThreads(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs && ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(std::forward<TArgs>(args)...)
            {
                static_assert(
                    dim::Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            TaskKernelCpuThreads(TaskKernelCpuThreads const &) = default;
            //-----------------------------------------------------------------------------
            TaskKernelCpuThreads(TaskKernelCpuThreads &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuThreads const &) -> TaskKernelCpuThreads & = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelCpuThreads &&) -> TaskKernelCpuThreads & = default;
            //-----------------------------------------------------------------------------
            ~TaskKernelCpuThreads() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            ALPAKA_FN_HOST auto operator()() const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const gridBlockExtent(
                    workdiv::getWorkDiv<Grid, Blocks>(*this));
                auto const blockThreadExtent(
                    workdiv::getWorkDiv<Block, Threads>(*this));
                auto const threadElemExtent(
                    workdiv::getWorkDiv<Thread, Elems>(*this));

                // Get the size of the block shared dynamic memory.
                auto const blockSharedMemDynSizeBytes(
                    meta::apply(
                        [&](std::decay_t<TArgs> const & ... args)
                        {
                            return
                                kernel::getBlockSharedMemDynSizeBytes<
                                    acc::AccCpuThreads<TDim, TIdx>>(
                                        m_kernelFnObj,
                                        blockThreadExtent,
                                        threadElemExtent,
                                        args...);
                        },
                        m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__
                    << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif
                acc::AccCpuThreads<TDim, TIdx> acc(
                    *static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this),
                    blockSharedMemDynSizeBytes);

                auto const blockThreadCount(blockThreadExtent.prod());
                ThreadPool threadPool(blockThreadCount);

                // Bind the kernel and its arguments to the grid block function.
                auto const boundGridBlockExecHost(
                    meta::apply(
                        [this, &acc, &blockThreadExtent, &threadPool](std::decay_t<TArgs> const & ... args)
                        {
                            return
                                std::bind(
                                    &TaskKernelCpuThreads::gridBlockExecHost,
                                    std::ref(acc),
                                    std::placeholders::_1,
                                    std::ref(blockThreadExtent),
                                    std::ref(threadPool),
                                    std::ref(m_kernelFnObj),
                                    std::ref(args)...);
                        },
                        m_args));

                // Execute the blocks serially.
                meta::ndLoopIncIdx(
                    gridBlockExtent,
                    boundGridBlockExecHost);
            }

        private:
            //-----------------------------------------------------------------------------
            //! The function executed for each grid block.
            ALPAKA_FN_HOST static auto gridBlockExecHost(
                acc::AccCpuThreads<TDim, TIdx> & acc,
                vec::Vec<TDim, TIdx> const & gridBlockIdx,
                vec::Vec<TDim, TIdx> const & blockThreadExtent,
                ThreadPool & threadPool,
                TKernelFnObj const & kernelFnObj,
                std::decay_t<TArgs> const & ... args)
            -> void
            {
                    // The futures of the threads in the current block.
                std::vector<std::future<void>> futuresInBlock;

                // Set the index of the current block
                acc.m_gridBlockIdx = gridBlockIdx;

                // Bind the kernel and its arguments to the host block thread execution function.
                auto boundBlockThreadExecHost(std::bind(
                    &TaskKernelCpuThreads::blockThreadExecHost,
                    std::ref(acc),
                    std::ref(futuresInBlock),
                    std::placeholders::_1,
                    std::ref(threadPool),
                    std::ref(kernelFnObj),
                    std::ref(args)...));
                // Execute the block threads in parallel.
                meta::ndLoopIncIdx(
                    blockThreadExtent,
                    boundBlockThreadExecHost);
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                // Wait for the completion of the block thread kernels.
                std::for_each(
                    futuresInBlock.begin(),
                    futuresInBlock.end(),
                    [](std::future<void> & t)
                    {
                        t.wait();
                    }
                );
#endif
                // Clean up.
                futuresInBlock.clear();

                acc.m_threadToIndexMap.clear();

                // After a block has been processed, the shared memory has to be deleted.
                block::shared::st::freeMem(acc);
            }
            //-----------------------------------------------------------------------------
            //! The function executed for each block thread on the host.
            ALPAKA_FN_HOST static auto blockThreadExecHost(
                acc::AccCpuThreads<TDim, TIdx> & acc,
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                std::vector<std::future<void>> & futuresInBlock,
                vec::Vec<TDim, TIdx> const & blockThreadIdx,
                ThreadPool & threadPool,
#else
                std::vector<std::future<void>> &,
                vec::Vec<TDim, TIdx> const & blockThreadIdx,
                ThreadPool &,
#endif
                TKernelFnObj const & kernelFnObj,
                std::decay_t<TArgs> const & ... args)
            -> void
            {
                // Bind the arguments to the accelerator block thread execution function.
                // The blockThreadIdx is required to be copied in because the variable will get changed for the next iteration/thread.
                auto boundBlockThreadExecAcc(
                    [&, blockThreadIdx]()
                    {
                        blockThreadExecAcc(
                            acc,
                            blockThreadIdx,
                            kernelFnObj,
                            args...);
                    });
                // Add the bound function to the block thread pool.
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                futuresInBlock.emplace_back(
                    threadPool.enqueueTask(
                        boundBlockThreadExecAcc));
#else
                (void)boundBlockThreadExecAcc;
#endif
            }
            //-----------------------------------------------------------------------------
            //! The thread entry point on the accelerator.
            ALPAKA_FN_HOST static auto blockThreadExecAcc(
                acc::AccCpuThreads<TDim, TIdx> & acc,
                vec::Vec<TDim, TIdx> const & blockThreadIdx,
                TKernelFnObj const & kernelFnObj,
                std::decay_t<TArgs> const & ... args)
            -> void
            {
                // We have to store the thread data before the kernel is calling any of the methods of this class depending on them.
                auto const threadId(std::this_thread::get_id());

                // Set the master thread id.
                if(blockThreadIdx.sum() == 0)
                {
                    acc.m_idMasterThread = threadId;
                }

                {
                    // The insertion of elements has to be done one thread at a time.
                    std::lock_guard<std::mutex> lock(acc.m_mtxMapInsert);

                    // Save the thread id, and index.
                    acc.m_threadToIndexMap.emplace(threadId, blockThreadIdx);
                }

                // Sync all threads so that the maps with thread id's are complete and not changed after here.
                syncBlockThreads(acc);

                // Execute the kernel itself.
                kernelFnObj(
                    const_cast<acc::AccCpuThreads<TDim, TIdx> const &>(acc),
                    args...);

                // We have to sync all threads here because if a thread would finish before all threads have been started,
                // a new thread could get the recycled (then duplicate) thread id!
                syncBlockThreads(acc);
            }

            TKernelFnObj m_kernelFnObj;
            std::tuple<std::decay_t<TArgs>...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads execution task accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                kernel::TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuThreads<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads execution task device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                kernel::TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevCpu;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads execution task dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                kernel::TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                kernel::TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads execution task idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                kernel::TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
