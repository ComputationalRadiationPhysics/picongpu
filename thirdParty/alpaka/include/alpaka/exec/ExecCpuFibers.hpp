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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccCpuFibers.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Fibers.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>
#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/ApplyTuple.hpp>

#include <algorithm>
#include <vector>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace exec
    {
        //#############################################################################
        //! The CPU fibers accelerator executor.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuFibers final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        private:
            //#############################################################################
            //! The type given to the ConcurrentExecPool for yielding the current fiber.
            struct FiberPoolYield
            {
                //-----------------------------------------------------------------------------
                //! Yields the current fiber.
                ALPAKA_FN_HOST static auto yield()
                -> void
                {
                    boost::this_fiber::yield();
                }
            };
            //#############################################################################
            // Yielding is not faster for fibers. Therefore we use condition variables.
            // It is better to wake them up when the conditions are fulfilled because this does not cost as much as for real threads.
            using FiberPool = alpaka::core::detail::ConcurrentExecPool<
                TIdx,
                boost::fibers::fiber,               // The concurrent execution type.
                boost::fibers::promise,             // The promise type.
                FiberPoolYield,                     // The type yielding the current concurrent execution.
                boost::fibers::mutex,               // The mutex type to use. Only required if TisYielding is true.
                boost::fibers::condition_variable,  // The condition variable type to use. Only required if TisYielding is true.
                false>;                             // If the threads should yield.

        public:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuFibers(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            ExecCpuFibers(ExecCpuFibers const &) = default;
            //-----------------------------------------------------------------------------
            ExecCpuFibers(ExecCpuFibers &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuFibers const &) -> ExecCpuFibers & = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecCpuFibers &&) -> ExecCpuFibers & = default;
            //-----------------------------------------------------------------------------
            ~ExecCpuFibers() = default;

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
                                    acc::AccCpuFibers<TDim, TIdx>>(
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
                acc::AccCpuFibers<TDim, TIdx> acc(
                    *static_cast<workdiv::WorkDivMembers<TDim, TIdx> const *>(this),
                    blockSharedMemDynSizeBytes);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << BOOST_CURRENT_FUNCTION
                    << " Fiber stack idx: " << boost::fibers::fixedsize_stack::traits_type::default_size() << " B" << std::endl;
#endif

                auto const blockThreadCount(blockThreadExtent.prod());
                FiberPool fiberPool(blockThreadCount);

                auto const boundGridBlockExecHost(
                    meta::apply(
                        [this, &acc, &blockThreadExtent, &fiberPool](TArgs const & ... args)
                        {
                            // Bind the kernel and its arguments to the grid block function.
                            return
                                std::bind(
                                    &ExecCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>::gridBlockExecHost,
                                    std::ref(acc),
                                    std::placeholders::_1,
                                    std::ref(blockThreadExtent),
                                    std::ref(fiberPool),
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
                acc::AccCpuFibers<TDim, TIdx> & acc,
                vec::Vec<TDim, TIdx> const & gridBlockIdx,
                vec::Vec<TDim, TIdx> const & blockThreadExtent,
                FiberPool & fiberPool,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args)
            -> void
            {
                    // The futures of the threads in the current block.
                std::vector<boost::fibers::future<void>> futuresInBlock;

                // Set the index of the current block
                acc.m_gridBlockIdx = gridBlockIdx;

                // Bind the kernel and its arguments to the host block thread execution function.
                auto boundBlockThreadExecHost(std::bind(
                    &ExecCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>::blockThreadExecHost,
                    std::ref(acc),
                    std::ref(futuresInBlock),
                    std::placeholders::_1,
                    std::ref(fiberPool),
                    std::ref(kernelFnObj),
                    std::ref(args)...));
                // Execute the block threads in parallel.
                meta::ndLoopIncIdx(
                    blockThreadExtent,
                    boundBlockThreadExecHost);

                // Wait for the completion of the block thread kernels.
                std::for_each(
                    futuresInBlock.begin(),
                    futuresInBlock.end(),
                    [](boost::fibers::future<void> & t)
                    {
                        t.wait();
                    }
                );
                // Clean up.
                futuresInBlock.clear();

                acc.m_fibersToIndices.clear();

                // After a block has been processed, the shared memory has to be deleted.
                block::shared::st::freeMem(acc);
            }
            //-----------------------------------------------------------------------------
            //! The function executed for each block thread.
            ALPAKA_FN_HOST static auto blockThreadExecHost(
                acc::AccCpuFibers<TDim, TIdx> & acc,
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                std::vector<boost::fibers::future<void>> & futuresInBlock,
                vec::Vec<TDim, TIdx> const & blockThreadIdx,
                FiberPool & fiberPool,
#else
                std::vector<boost::fibers::future<void>> &,
                vec::Vec<TDim, TIdx> const & blockThreadIdx,
                FiberPool &,
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
                        blockThreadFiberFn(
                            acc,
                            blockThreadIdx,
                            kernelFnObj,
                            args...);
                    });
                // Add the bound function to the block thread pool.
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
                futuresInBlock.emplace_back(
                    fiberPool.enqueueTask(
                        boundBlockThreadExecAcc));
#else
                (void)boundBlockThreadExecAcc;
#endif
            }
            //-----------------------------------------------------------------------------
            //! The fiber entry point.
            ALPAKA_FN_HOST static auto blockThreadFiberFn(
                acc::AccCpuFibers<TDim, TIdx> & acc,
                vec::Vec<TDim, TIdx> const & blockThreadIdx,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args)
            -> void
            {
                // We have to store the fiber data before the kernel is calling any of the methods of this class depending on them.
                auto const fiberId(boost::this_fiber::get_id());

                // Set the master thread id.
                if(blockThreadIdx.sum() == 0)
                {
                    acc.m_masterFiberId = fiberId;
                }

                // Save the fiber id, and index.
                acc.m_fibersToIndices.emplace(fiberId, blockThreadIdx);

                // Sync all threads so that the maps with thread id's are complete and not changed after here.
                syncBlockThreads(acc);

                // Execute the kernel itself.
                kernelFnObj(
                    const_cast<acc::AccCpuFibers<TDim, TIdx> const &>(acc),
                    args...);

                // We have to sync all fibers here because if a fiber would finish before all fibers have been started, the new fiber could get a recycled (then duplicate) fiber id!
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
            //! The CPU fibers executor accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuFibers<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers executor device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU fibers executor dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU fibers executor platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                exec::ExecCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU fibers executor idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                exec::ExecCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
