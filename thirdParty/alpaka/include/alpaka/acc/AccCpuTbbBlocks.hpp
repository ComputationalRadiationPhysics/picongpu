/* Copyright 2026 Axel Huebl, Benjamin Worpitz, Erik Zenker, René Widera, Jan Stephan, Bernhard Manfred Gruber,
 *                Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "alpaka/atomic/AtomicCpu.hpp"
#include "alpaka/atomic/AtomicHierarchy.hpp"
#include "alpaka/atomic/AtomicNoOp.hpp"
#include "alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp"
#include "alpaka/block/shared/st/BlockSharedMemStMember.hpp"
#include "alpaka/block/sync/BlockSyncNoOp.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/grid/GridSyncCpuTbbBlocks.hpp"
#include "alpaka/idx/bt/IdxBtZero.hpp"
#include "alpaka/idx/gb/IdxGbRef.hpp"
#include "alpaka/intrinsic/IntrinsicCpu.hpp"
#include "alpaka/math/MathStdLib.hpp"
#include "alpaka/mem/fence/MemFenceCpu.hpp"
#include "alpaka/rand/RandDefault.hpp"
#include "alpaka/rand/RandStdLib.hpp"
#include "alpaka/warp/WarpSingleThread.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

// Specialized traits.
#include "alpaka/acc/Traits.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/Traits.hpp"

// Implementation details.
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/BarrierTbb.h"
#include "alpaka/core/ClipCast.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/dev/DevCpu.hpp"

#ifdef __cpp_lib_format
#    include <format>
#endif
#include <string>
#include <typeinfo>

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#    include <tbb/tbb.h>

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuTbbBlocks;

    //! The CPU TBB block accelerator.
    template<typename TDim, typename TIdx>
    class AccCpuTbbBlocks final
        : public WorkDivMembers<TDim, TIdx>
        , public gb::IdxGbRef<TDim, TIdx>
        , public bt::IdxBtZero<TDim, TIdx>
        , public AtomicHierarchy<
              AtomicCpu, // grid atomics
              AtomicCpu, // block atomics
              AtomicNoOp> // thread atomics
        , public math::MathStdLib
        , public BlockSharedMemDynMember<>
        , public BlockSharedMemStMember<>
        , public BlockSyncNoOp
        , public GridSyncBarrierTbb<TIdx>
        , public IntrinsicCpu
        , public MemFenceCpu
#    ifdef ALPAKA_DISABLE_VENDOR_RNG
        , public rand::RandDefault
#    else
        , public rand::RandStdLib
#    endif
        , public warp::WarpSingleThread
        , public interface::Implements<InterfaceAcc, AccCpuTbbBlocks<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelCpuTbbBlocks;

        AccCpuTbbBlocks(AccCpuTbbBlocks const&) = delete;
        AccCpuTbbBlocks(AccCpuTbbBlocks&&) = delete;
        auto operator=(AccCpuTbbBlocks const&) -> AccCpuTbbBlocks& = delete;
        auto operator=(AccCpuTbbBlocks&&) -> AccCpuTbbBlocks& = delete;

    private:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST AccCpuTbbBlocks(
            TWorkDiv const& workDiv,
            std::size_t const& blockSharedMemDynSizeBytes,
            core::tbb::BarrierThread<TIdx>& barrier)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
            , BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
            , BlockSharedMemStMember<>(staticMemBegin(), staticMemCapacity())
            , GridSyncBarrierTbb<TIdx>(barrier)
            , m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
        {
        }

    private:
        // getIdx
        Vec<TDim, TIdx> mutable m_gridBlockIdx; //!< The index of the currently executed block.
    };

    namespace trait
    {
        //! The CPU TBB block accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = AccCpuTbbBlocks<TDim, TIdx>;
        };

        //! The CPU TBB block single thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsSingleThreadAcc<AccCpuTbbBlocks<TDim, TIdx>> : std::true_type
        {
        };

        //! The CPU TBB block multi thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsMultiThreadAcc<AccCpuTbbBlocks<TDim, TIdx>> : std::false_type
        {
        };

        //! The CPU TBB block accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccCpuTbbBlocks<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> AccDevProps<TDim, TIdx>
            {
                return {// m_multiProcessorCount
                        alpaka::core::clipCast<TIdx>(tbb::this_task_arena::max_concurrency()),
                        // m_gridBlockExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        Vec<TDim, TIdx>::ones(),
                        // m_blockThreadCountMax
                        static_cast<TIdx>(1),
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(AccCpuTbbBlocks<TDim, TIdx>::staticAllocBytes()),
                        // m_globalMemSizeBytes
                        getMemBytes(dev),
                        // m_cooperativeLaunch
                        true};
            }
        };

        //! The CPU TBB block accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccCpuTbbBlocks<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
#    if ALPAKA_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wexit-time-destructors"
#    endif
                using namespace std::literals;
                static std::string const accName =
#    ifdef __cpp_lib_format
                    std::format("AccCpuTbbBlocks<{},{}>", TDim::value, core::demangled<TIdx>);
#    else
                    "AccCpuTbbBlocks<"s + std::to_string(TDim::value) + ","s + std::string(core::demangled<TIdx>)
                    + ">"s;
#    endif
                return accName;
#    if ALPAKA_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
            }
        };

        //! The CPU TBB block accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = DevCpu;
        };

        //! The CPU TBB block accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The CPU TBB block accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccCpuTbbBlocks<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                if(workDiv.m_blockThreadExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error(
                        "The given work division is not valid for a single thread Acc: "
                        + getAccName<AccCpuTbbBlocks<TDim, TIdx>>() + ". Threads per block should be 1!");
                }

                return TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU TBB block accelerator execution cooperative task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskCooperativeKernel<AccCpuTbbBlocks<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            ALPAKA_FN_HOST static auto createTaskCooperativeKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                if(workDiv.m_blockThreadExtent.prod() != static_cast<TIdx>(1u))
                {
                    throw std::runtime_error(
                        "The given work division is not valid for a single thread Acc: "
                        + getAccName<AccCpuTbbBlocks<TDim, TIdx>>() + ". Threads per block should be 1!");
                }
                auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(workDiv);
                auto const maxBlocks = tbb::this_task_arena::max_concurrency();
                if(gridBlockExtent.prod() > static_cast<TIdx>(maxBlocks))
                {
                    throw std::runtime_error(
                        "The number of requested blocks is larger than maximuma of the device for TBB "
                        "accelerator. Requested: "
                        + std::to_string(gridBlockExtent.prod()) + ", maximum allowed: " + std::to_string(maxBlocks)
                        + ". Use getMaxActiveBlocks().");
                }

                return TaskKernelCpuTbbBlocks<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU TBB block execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PlatformType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = PlatformCpu;
        };

        //! The CPU TBB block accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = TIdx;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccCpuTbbBlocks<TDim, TIdx>>
        {
            using type = alpaka::TagCpuTbbBlocks;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagCpuTbbBlocks, TDim, TIdx>
        {
            using type = alpaka::AccCpuTbbBlocks<TDim, TIdx>;
        };
    } // namespace trait
} // namespace alpaka

#endif
