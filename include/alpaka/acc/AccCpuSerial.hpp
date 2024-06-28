/* Copyright 2024 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
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
#include "alpaka/idx/bt/IdxBtZero.hpp"
#include "alpaka/idx/gb/IdxGbRef.hpp"
#include "alpaka/intrinsic/IntrinsicCpu.hpp"
#include "alpaka/math/MathStdLib.hpp"
#include "alpaka/mem/fence/MemFenceCpuSerial.hpp"
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
#include "alpaka/core/Concepts.hpp"
#include "alpaka/dev/DevCpu.hpp"

#include <memory>
#include <typeinfo>

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuSerial;

    //! The CPU serial accelerator.
    //!
    //! This accelerator allows serial kernel execution on a CPU device.
    //! The block idx is restricted to 1x1x1 and all blocks are executed serially so there is no parallelism at all.
    template<typename TDim, typename TIdx>
    class AccCpuSerial final
        : public WorkDivMembers<TDim, TIdx>
        , public gb::IdxGbRef<TDim, TIdx>
        , public bt::IdxBtZero<TDim, TIdx>
        , public AtomicHierarchy<
              AtomicCpu, // grid atomics
              AtomicNoOp, // block atomics
              AtomicNoOp> // thread atomics
        , public math::MathStdLib
        , public BlockSharedMemDynMember<>
        , public BlockSharedMemStMember<>
        , public BlockSyncNoOp
        , public IntrinsicCpu
        , public MemFenceCpuSerial
#    ifdef ALPAKA_DISABLE_VENDOR_RNG
        , public rand::RandDefault
#    else
        , public rand::RandStdLib
#    endif
        , public warp::WarpSingleThread
        , public concepts::Implements<ConceptAcc, AccCpuSerial<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelCpuSerial;

        AccCpuSerial(AccCpuSerial const&) = delete;
        AccCpuSerial(AccCpuSerial&&) = delete;
        auto operator=(AccCpuSerial const&) -> AccCpuSerial& = delete;
        auto operator=(AccCpuSerial&&) -> AccCpuSerial& = delete;

    private:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST AccCpuSerial(TWorkDiv const& workDiv, size_t const& blockSharedMemDynSizeBytes)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
            , BlockSharedMemDynMember<>(blockSharedMemDynSizeBytes)
            , BlockSharedMemStMember<>(staticMemBegin(), staticMemCapacity())
            , m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
        {
        }

    private:
        // getIdx
        Vec<TDim, TIdx> mutable m_gridBlockIdx; //!< The index of the currently executed block.
    };

    namespace trait
    {
        //! The CPU serial accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccCpuSerial<TDim, TIdx>>
        {
            using type = AccCpuSerial<TDim, TIdx>;
        };

        //! The CPU serial single thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsSingleThreadAcc<AccCpuSerial<TDim, TIdx>> : std::true_type
        {
        };

        //! The CPU serial multi thread accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct IsMultiThreadAcc<AccCpuSerial<TDim, TIdx>> : std::false_type
        {
        };

        //! The CPU serial accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccCpuSerial<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> AccDevProps<TDim, TIdx>
            {
                return {// m_multiProcessorCount
                        static_cast<TIdx>(1),
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
                        static_cast<size_t>(AccCpuSerial<TDim, TIdx>::staticAllocBytes()),
                        // m_globalMemSizeBytes
                        getMemBytes(dev)};
            }
        };

        //! The CPU serial accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccCpuSerial<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccCpuSerial<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
            }
        };

        //! The CPU serial accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccCpuSerial<TDim, TIdx>>
        {
            using type = DevCpu;
        };

        //! The CPU serial accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccCpuSerial<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The CPU serial accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccCpuSerial<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelCpuSerial<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU serial execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PlatformType<AccCpuSerial<TDim, TIdx>>
        {
            using type = PlatformCpu;
        };

        //! The CPU serial accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccCpuSerial<TDim, TIdx>>
        {
            using type = TIdx;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccCpuSerial<TDim, TIdx>>
        {
            using type = alpaka::TagCpuSerial;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagCpuSerial, TDim, TIdx>
        {
            using type = alpaka::AccCpuSerial<TDim, TIdx>;
        };
    } // namespace trait
} // namespace alpaka

#endif
