/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Specialized traits.
#include "alpaka/acc/Traits.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/platform/Traits.hpp"

// Implementation details.
#include "alpaka/acc/AccCpuSerial.hpp"
#include "alpaka/core/Decay.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    include <iostream>
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

namespace alpaka
{
    //! The CPU serial execution task implementation.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuSerial final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelCpuSerial(TWorkDiv&& workDiv, TKernelFnObj kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(std::move(kernelFnObj))
            , m_args(std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        //! Executes the kernel function object.
        ALPAKA_FN_HOST auto operator()() const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(*this);
            auto const blockThreadExtent = getWorkDiv<Block, Threads>(*this);
            auto const threadElemExtent = getWorkDiv<Thread, Elems>(*this);

            // Get the size of the block shared dynamic memory.
            auto const blockSharedMemDynSizeBytes = std::apply(
                [&](std::decay_t<TArgs> const&... args)
                {
                    return getBlockSharedMemDynSizeBytes<AccCpuSerial<TDim, TIdx>>(
                        m_kernelFnObj,
                        blockThreadExtent,
                        threadElemExtent,
                        args...);
                },
                m_args);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " blockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                      << std::endl;
#    endif

            AccCpuSerial<TDim, TIdx> acc(
                *static_cast<WorkDivMembers<TDim, TIdx> const*>(this),
                blockSharedMemDynSizeBytes);

            if(blockThreadExtent.prod() != static_cast<TIdx>(1u))
            {
                throw std::runtime_error("A block for the serial accelerator can only ever have one single thread!");
            }

            // Execute the blocks serially.
            meta::ndLoopIncIdx(
                gridBlockExtent,
                [&](Vec<TDim, TIdx> const& blockThreadIdx)
                {
                    acc.m_gridBlockIdx = blockThreadIdx;

                    std::apply(m_kernelFnObj, std::tuple_cat(std::tie(acc), m_args));

                    // After a block has been processed, the shared memory has to be deleted.
                    freeSharedVars(acc);
                });
        }

    private:
        TKernelFnObj m_kernelFnObj;
        std::tuple<std::decay_t<TArgs>...> m_args;
    };

    namespace trait
    {
        //! The CPU serial execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccCpuSerial<TDim, TIdx>;
        };

        //! The CPU serial execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevCpu;
        };

        //! The CPU serial execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //! The CPU serial execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PlatformType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PlatformCpu;
        };

        //! The CPU serial execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka

#endif
