/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

// Implementation details.
#    include <alpaka/acc/AccCpuFibers.hpp>
#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/ConcurrentExecPool.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/core/Fibers.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/meta/ApplyTuple.hpp>
#    include <alpaka/meta/NdLoop.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    include <algorithm>
#    include <functional>
#    include <tuple>
#    include <type_traits>
#    include <vector>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

namespace alpaka
{
    //#############################################################################
    //! The CPU fibers accelerator execution task.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuFibers final : public WorkDivMembers<TDim, TIdx>
    {
    private:
        //#############################################################################
        //! The type given to the ConcurrentExecPool for yielding the current fiber.
        struct FiberPoolYield
        {
            //-----------------------------------------------------------------------------
            //! Yields the current fiber.
            ALPAKA_FN_HOST static auto yield() -> void
            {
                boost::this_fiber::yield();
            }
        };
        //#############################################################################
        // Yielding is not faster for fibers. Therefore we use condition variables.
        // It is better to wake them up when the conditions are fulfilled because this does not cost as much as for
        // real threads.
        using FiberPool = alpaka::core::detail::ConcurrentExecPool<
            TIdx,
            boost::fibers::fiber, // The concurrent execution type.
            boost::fibers::promise, // The promise type.
            FiberPoolYield, // The type yielding the current concurrent execution.
            boost::fibers::mutex, // The mutex type to use. Only required if TisYielding is true.
            boost::fibers::condition_variable, // The condition variable type to use. Only required if TisYielding is
                                               // true.
            false>; // If the threads should yield.

    public:
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelCpuFibers(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(kernelFnObj)
            , m_args(std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }
        //-----------------------------------------------------------------------------
        TaskKernelCpuFibers(TaskKernelCpuFibers const&) = default;
        //-----------------------------------------------------------------------------
        TaskKernelCpuFibers(TaskKernelCpuFibers&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelCpuFibers const&) -> TaskKernelCpuFibers& = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelCpuFibers&&) -> TaskKernelCpuFibers& = default;
        //-----------------------------------------------------------------------------
        ~TaskKernelCpuFibers() = default;

        //-----------------------------------------------------------------------------
        //! Executes the kernel function object.
        ALPAKA_FN_HOST auto operator()() const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            auto const gridBlockExtent(getWorkDiv<Grid, Blocks>(*this));
            auto const blockThreadExtent(getWorkDiv<Block, Threads>(*this));
            auto const threadElemExtent(getWorkDiv<Thread, Elems>(*this));

            // Get the size of the block shared dynamic memory.
            auto const blockSharedMemDynSizeBytes(meta::apply(
                [&](ALPAKA_DECAY_T(TArgs) const&... args) {
                    return getBlockSharedMemDynSizeBytes<AccCpuFibers<TDim, TIdx>>(
                        m_kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...);
                },
                m_args));

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                      << std::endl;
#    endif
            AccCpuFibers<TDim, TIdx> acc(
                *static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
                blockSharedMemDynSizeBytes);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__
                      << " Fiber stack idx: " << boost::fibers::fixedsize_stack::traits_type::default_size() << " B"
                      << std::endl;
#    endif

            auto const blockThreadCount(blockThreadExtent.prod());
            FiberPool fiberPool(blockThreadCount);

            auto const boundGridBlockExecHost(meta::apply(
                [this, &acc, &blockThreadExtent, &fiberPool](ALPAKA_DECAY_T(TArgs) const&... args) {
                    // Bind the kernel and its arguments to the grid block function.
                    return std::bind(
                        &TaskKernelCpuFibers::gridBlockExecHost,
                        std::ref(acc),
                        std::placeholders::_1,
                        std::ref(blockThreadExtent),
                        std::ref(fiberPool),
                        std::ref(m_kernelFnObj),
                        std::ref(args)...);
                },
                m_args));

            // Execute the blocks serially.
            meta::ndLoopIncIdx(gridBlockExtent, boundGridBlockExecHost);
        }

    private:
        //-----------------------------------------------------------------------------
        //! The function executed for each grid block.
        ALPAKA_FN_HOST static auto gridBlockExecHost(
            AccCpuFibers<TDim, TIdx>& acc,
            Vec<TDim, TIdx> const& gridBlockIdx,
            Vec<TDim, TIdx> const& blockThreadExtent,
            FiberPool& fiberPool,
            TKernelFnObj const& kernelFnObj,
            std::decay_t<TArgs> const&... args) -> void
        {
            // The futures of the threads in the current block.
            std::vector<boost::fibers::future<void>> futuresInBlock;

            // Set the index of the current block
            acc.m_gridBlockIdx = gridBlockIdx;

            // Bind the kernel and its arguments to the host block thread execution function.
            auto boundBlockThreadExecHost(std::bind(
                &TaskKernelCpuFibers::blockThreadExecHost,
                std::ref(acc),
                std::ref(futuresInBlock),
                std::placeholders::_1,
                std::ref(fiberPool),
                std::ref(kernelFnObj),
                std::ref(args)...));
            // Execute the block threads in parallel.
            meta::ndLoopIncIdx(blockThreadExtent, boundBlockThreadExecHost);

            // Wait for the completion of the block thread kernels.
            std::for_each(futuresInBlock.begin(), futuresInBlock.end(), [](boost::fibers::future<void>& t) {
                t.wait();
            });
            // Clean up.
            futuresInBlock.clear();

            acc.m_fibersToIndices.clear();

            // After a block has been processed, the shared memory has to be deleted.
            freeSharedVars(acc);
        }
        //-----------------------------------------------------------------------------
        //! The function executed for each block thread.
        ALPAKA_FN_HOST static auto blockThreadExecHost(
            AccCpuFibers<TDim, TIdx>& acc,
#    if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
            std::vector<boost::fibers::future<void>>& futuresInBlock,
            Vec<TDim, TIdx> const& blockThreadIdx,
            FiberPool& fiberPool,
#    else
            std::vector<boost::fibers::future<void>>&,
            Vec<TDim, TIdx> const& blockThreadIdx,
            FiberPool&,
#    endif
            TKernelFnObj const& kernelFnObj,
            std::decay_t<TArgs> const&... args) -> void
        {
            // Bind the arguments to the accelerator block thread execution function.
            // The blockThreadIdx is required to be copied in because the variable will get changed for the next
            // iteration/thread.
            auto boundBlockThreadExecAcc(
                [&, blockThreadIdx]() { blockThreadFiberFn(acc, blockThreadIdx, kernelFnObj, args...); });
            // Add the bound function to the block thread pool.
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#    if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
            futuresInBlock.emplace_back(fiberPool.enqueueTask(boundBlockThreadExecAcc));
#    else
            (void) boundBlockThreadExecAcc;
#    endif
        }
        //-----------------------------------------------------------------------------
        //! The fiber entry point.
        ALPAKA_FN_HOST static auto blockThreadFiberFn(
            AccCpuFibers<TDim, TIdx>& acc,
            Vec<TDim, TIdx> const& blockThreadIdx,
            TKernelFnObj const& kernelFnObj,
            std::decay_t<TArgs> const&... args) -> void
        {
            // We have to store the fiber data before the kernel is calling any of the methods of this class depending
            // on them.
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
            kernelFnObj(const_cast<AccCpuFibers<TDim, TIdx> const&>(acc), args...);

            // We have to sync all fibers here because if a fiber would finish before all fibers have been started, the
            // new fiber could get a recycled (then duplicate) fiber id!
            syncBlockThreads(acc);
        }

        TKernelFnObj m_kernelFnObj;
        std::tuple<std::decay_t<TArgs>...> m_args;
    };

    namespace traits
    {
        //#############################################################################
        //! The CPU fibers execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccCpuFibers<TDim, TIdx>;
        };

        //#############################################################################
        //! The CPU fibers execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevCpu;
        };

        //#############################################################################
        //! The CPU fibers execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The CPU fibers execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfCpu;
        };

        //#############################################################################
        //! The CPU fibers execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelCpuFibers<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#endif
