/* Copyright 2024 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "alpaka/atomic/AtomicCpu.hpp"
#include "alpaka/atomic/AtomicHierarchy.hpp"
#include "alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp"
#include "alpaka/block/shared/st/BlockSharedMemStMemberMasterSync.hpp"
#include "alpaka/block/sync/BlockSyncBarrierThread.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/idx/bt/IdxBtRefThreadIdMap.hpp"
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
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/ClipCast.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/dev/DevCpu.hpp"

#include <memory>
#include <thread>
#include <typeinfo>

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuThreads;

    //! The CPU threads accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a CPU device.
    //! It uses std::thread to implement the parallelism.
    template<typename TDim, typename TIdx>
    class AccCpuThreads final
        : public WorkDivMembers<TDim, TIdx>
        , public gb::IdxGbRef<TDim, TIdx>
        , public bt::IdxBtRefThreadIdMap<TDim, TIdx>
        , public AtomicHierarchy<
              AtomicCpu, // grid atomics
              AtomicCpu, // block atomics
              AtomicCpu> // thread atomics
        , public math::MathStdLib
        , public BlockSharedMemDynMember<>
        , public BlockSharedMemStMemberMasterSync<>
        , public BlockSyncBarrierThread<TIdx>
        , public IntrinsicCpu
        , public MemFenceCpu
#    ifdef ALPAKA_DISABLE_VENDOR_RNG
        , public rand::RandDefault
#    else
        , public rand::RandStdLib
#    endif
        , public warp::WarpSingleThread
        , public concepts::Implements<ConceptAcc, AccCpuThreads<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelCpuThreads;

        AccCpuThreads(AccCpuThreads const&) = delete;
        AccCpuThreads(AccCpuThreads&&) = delete;
        auto operator=(AccCpuThreads const&) -> AccCpuThreads& = delete;
        auto operator=(AccCpuThreads&&) -> AccCpuThreads& = delete;

    private:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST AccCpuThreads(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
            , bt::IdxBtRefThreadIdMap<TDim, TIdx>(m_threadToIndexMap)
            , BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
            , BlockSharedMemStMemberMasterSync<>(
                  staticMemBegin(),
                  staticMemCapacity(),
                  [this]() { syncBlockThreads(*this); },
                  [this]() noexcept { return (m_idMasterThread == std::this_thread::get_id()); })
            , BlockSyncBarrierThread<TIdx>(getWorkDiv<Block, Threads>(workDiv).prod())
            , m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
        {
        }

    private:
        // getIdx
        std::mutex mutable m_mtxMapInsert; //!< The mutex used to secure insertion into the ThreadIdToIdxMap.
        typename bt::IdxBtRefThreadIdMap<TDim, TIdx>::
            ThreadIdToIdxMap mutable m_threadToIndexMap; //!< The mapping of thread id's to indices.
        Vec<TDim, TIdx> mutable m_gridBlockIdx; //!< The index of the currently executed block.

        // allocBlockSharedArr
        std::thread::id mutable m_idMasterThread; //!< The id of the master thread.
    };

    namespace trait
    {
        //! The CPU threads accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccCpuThreads<TDim, TIdx>>
        {
            using type = AccCpuThreads<TDim, TIdx>;
        };

        //! The CPU threads single thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsSingleThreadAcc<AccCpuThreads<TDim, TIdx>> : std::false_type
        {
        };

        //! The CPU threads multi thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsMultiThreadAcc<AccCpuThreads<TDim, TIdx>> : std::true_type
        {
        };

        //! The CPU threads accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccCpuThreads<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> AccDevProps<TDim, TIdx>
            {
#    ifdef ALPAKA_CI
                auto const blockThreadCountMax = static_cast<TIdx>(8);
#    else
                // \TODO: Magic number. What is the maximum? Just set a reasonable value? There is a implementation
                // defined maximum where the creation of a new thread crashes. std::thread::hardware_concurrency can
                // return 0, so 1 is the default case?
                auto const blockThreadCountMax = std::max(
                    static_cast<TIdx>(1),
                    alpaka::core::clipCast<TIdx>(std::thread::hardware_concurrency() * 8));
#    endif
                auto const memBytes = getMemBytes(dev);
                return {// m_multiProcessorCount
                        static_cast<TIdx>(1),
                        // m_gridBlockExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        Vec<TDim, TIdx>::all(blockThreadCountMax),
                        // m_blockThreadCountMax
                        blockThreadCountMax,
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        memBytes,
                        // m_globalMemSizeBytes
                        memBytes};
            }
        };

        //! The CPU threads accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccCpuThreads<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccCpuThreads<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
            }
        };

        //! The CPU threads accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccCpuThreads<TDim, TIdx>>
        {
            using type = DevCpu;
        };

        //! The CPU threads accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccCpuThreads<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The CPU threads accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccCpuThreads<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU threads execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PlatformType<AccCpuThreads<TDim, TIdx>>
        {
            using type = PlatformCpu;
        };

        //! The CPU threads accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccCpuThreads<TDim, TIdx>>
        {
            using type = TIdx;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccCpuThreads<TDim, TIdx>>
        {
            using type = alpaka::TagCpuThreads;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagCpuThreads, TDim, TIdx>
        {
            using type = alpaka::AccCpuThreads<TDim, TIdx>;
        };
    } // namespace trait
} // namespace alpaka

#endif
