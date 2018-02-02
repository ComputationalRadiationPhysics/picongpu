/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/size/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuThreads.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/core/ConcurrentExecPool.hpp>
#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/ApplyTuple.hpp>

#include <boost/predef.h>

#include <algorithm>
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
    namespace exec
    {
        //#############################################################################
        //! The CPU threads executor.
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuThreads final :
            public workdiv::WorkDivMembers<TDim, TSize>
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
                TSize,
                std::thread,        // The concurrent execution type.
                std::promise,       // The promise type.
                ThreadPoolYield>;   // The type yielding the current concurrent execution.

        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuThreads(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TSize>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            ExecCpuThreads(ExecCpuThreads const &) = default;
            //-----------------------------------------------------------------------------
            ExecCpuThreads(ExecCpuThreads &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuThreads const &) -> ExecCpuThreads & = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuThreads &&) -> ExecCpuThreads & = default;
            //-----------------------------------------------------------------------------
            ~ExecCpuThreads() = default;

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
                        [&](TArgs const & ... args)
                        {
                            return
                                kernel::getBlockSharedMemDynSizeBytes<
                                    acc::AccCpuThreads<TDim, TSize>>(
                                        m_kernelFnObj,
                                        blockThreadExtent,
                                        threadElemExtent,
                                        args...);
                        },
                        m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << BOOST_CURRENT_FUNCTION
                    << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif
                acc::AccCpuThreads<TDim, TSize> acc(
                    *static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this),
                    blockSharedMemDynSizeBytes);

                auto const blockThreadCount(blockThreadExtent.prod());
                ThreadPool threadPool(blockThreadCount);

                // Bind the kernel and its arguments to the grid block function.
                auto const boundGridBlockExecHost(
                    meta::apply(
                        [this, &acc, &blockThreadExtent, &threadPool](TArgs const & ... args)
                        {
                            return
                                std::bind(
                                    &ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>::gridBlockExecHost,
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
                acc::AccCpuThreads<TDim, TSize> & acc,
                vec::Vec<TDim, TSize> const & gridBlockIdx,
                vec::Vec<TDim, TSize> const & blockThreadExtent,
                ThreadPool & threadPool,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args)
            -> void
            {
                    // The futures of the threads in the current block.
                std::vector<std::future<void>> futuresInBlock;

                // Set the index of the current block
                acc.m_gridBlockIdx = gridBlockIdx;

                // Bind the kernel and its arguments to the host block thread execution function.
                auto boundBlockThreadExecHost(std::bind(
                    &ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>::blockThreadExecHost,
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
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
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
                acc::AccCpuThreads<TDim, TSize> & acc,
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
                std::vector<std::future<void>> & futuresInBlock,
                vec::Vec<TDim, TSize> const & blockThreadIdx,
                ThreadPool & threadPool,
#else
                std::vector<std::future<void>> &,
                vec::Vec<TDim, TSize> const & blockThreadIdx,
                ThreadPool &,
#endif
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args)
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
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_CUDA_DEVICE)
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
                acc::AccCpuThreads<TDim, TSize> & acc,
                vec::Vec<TDim, TSize> const & blockThreadIdx,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args)
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
                    const_cast<acc::AccCpuThreads<TDim, TSize> const &>(acc),
                    args...);

                // We have to sync all threads here because if a thread would finish before all threads have been started,
                // a new thread could get the recycled (then duplicate) thread id!
                syncBlockThreads(acc);
            }

            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads executor accelerator type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuThreads<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads executor device type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU threads executor dimension getter trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace exec
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads executor executor type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads executor platform type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                exec::ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads executor size type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecCpuThreads<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
