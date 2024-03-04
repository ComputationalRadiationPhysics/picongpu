/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "alpaka/atomic/AtomicCpu.hpp"
#include "alpaka/atomic/AtomicHierarchy.hpp"
#include "alpaka/atomic/AtomicNoOp.hpp"
#include "alpaka/atomic/AtomicOmpBuiltIn.hpp"
#include "alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp"
#include "alpaka/block/shared/st/BlockSharedMemStMember.hpp"
#include "alpaka/block/sync/BlockSyncNoOp.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/idx/bt/IdxBtZero.hpp"
#include "alpaka/idx/gb/IdxGbRef.hpp"
#include "alpaka/intrinsic/IntrinsicCpu.hpp"
#include "alpaka/math/MathStdLib.hpp"
#include "alpaka/mem/fence/MemFenceOmp2Blocks.hpp"
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

#include <limits>
#include <typeinfo>

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuOmp2Blocks;

    //! The CPU OpenMP 2.0 block accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a CPU device.
    //! It uses OpenMP 2.0 to implement the grid block parallelism.
    //! The block idx is restricted to 1x1x1.
    template<typename TDim, typename TIdx>
    class AccCpuOmp2Blocks final
        : public WorkDivMembers<TDim, TIdx>
        , public gb::IdxGbRef<TDim, TIdx>
        , public bt::IdxBtZero<TDim, TIdx>
        , public AtomicHierarchy<
              AtomicCpu, // grid atomics
              AtomicOmpBuiltIn, // block atomics
              AtomicNoOp> // thread atomics
        , public math::MathStdLib
        , public BlockSharedMemDynMember<>
        , public BlockSharedMemStMember<>
        , public BlockSyncNoOp
        , public IntrinsicCpu
        , public MemFenceOmp2Blocks
#    ifdef ALPAKA_DISABLE_VENDOR_RNG
        , public rand::RandDefault
#    else
        , public rand::RandStdLib
#    endif
        , public warp::WarpSingleThread
        , public concepts::Implements<ConceptAcc, AccCpuOmp2Blocks<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelCpuOmp2Blocks;

        AccCpuOmp2Blocks(AccCpuOmp2Blocks const&) = delete;
        AccCpuOmp2Blocks(AccCpuOmp2Blocks&&) = delete;
        auto operator=(AccCpuOmp2Blocks const&) -> AccCpuOmp2Blocks& = delete;
        auto operator=(AccCpuOmp2Blocks&&) -> AccCpuOmp2Blocks& = delete;

    private:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST AccCpuOmp2Blocks(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
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
        //! The CPU OpenMP 2.0 block accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = AccCpuOmp2Blocks<TDim, TIdx>;
        };

        //! The CPU OpenMP 2.0 block accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& /* dev */) -> alpaka::AccDevProps<TDim, TIdx>
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
                        static_cast<size_t>(AccCpuOmp2Blocks<TDim, TIdx>::staticAllocBytes())};
            }
        };

        //! The CPU OpenMP 2.0 block accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccCpuOmp2Blocks<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
            }
        };

        //! The CPU OpenMP 2.0 block accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = DevCpu;
        };

        //! The CPU OpenMP 2.0 block accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The CPU OpenMP 2.0 block accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccCpuOmp2Blocks<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };

        //! The CPU OpenMP 2.0 block execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PlatformType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = PlatformCpu;
        };

        //! The CPU OpenMP 2.0 block accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = TIdx;
        };

        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccCpuOmp2Blocks<TDim, TIdx>>
        {
            using type = alpaka::TagCpuOmp2Blocks;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagCpuOmp2Blocks, TDim, TIdx>
        {
            using type = alpaka::AccCpuOmp2Blocks<TDim, TIdx>;
        };
    } // namespace trait
} // namespace alpaka

#endif
