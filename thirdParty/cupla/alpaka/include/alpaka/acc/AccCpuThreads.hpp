/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

// Base classes.
#    include <alpaka/atomic/AtomicHierarchy.hpp>
#    include <alpaka/atomic/AtomicStdLibLock.hpp>
#    include <alpaka/block/shared/dyn/BlockSharedMemDynAlignedAlloc.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStMasterSync.hpp>
#    include <alpaka/block/sync/BlockSyncBarrierThread.hpp>
#    include <alpaka/idx/bt/IdxBtRefThreadIdMap.hpp>
#    include <alpaka/idx/gb/IdxGbRef.hpp>
#    include <alpaka/intrinsic/IntrinsicCpu.hpp>
#    include <alpaka/math/MathStdLib.hpp>
#    include <alpaka/rand/RandStdLib.hpp>
#    include <alpaka/time/TimeStdLib.hpp>
#    include <alpaka/warp/WarpSingleThread.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>

// Implementation details.
#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/ClipCast.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/dev/DevCpu.hpp>

#    include <memory>
#    include <thread>
#    include <typeinfo>

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuThreads;

    //#############################################################################
    //! The CPU threads accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a CPU device.
    //! It uses std::thread to implement the parallelism.
    template<
        typename TDim,
        typename TIdx>
    class AccCpuThreads final :
        public WorkDivMembers<TDim, TIdx>,
        public gb::IdxGbRef<TDim, TIdx>,
        public bt::IdxBtRefThreadIdMap<TDim, TIdx>,
        public AtomicHierarchy<
            AtomicStdLibLock<16>, // grid atomics
            AtomicStdLibLock<16>, // block atomics
            AtomicStdLibLock<16>  // thread atomics
        >,
        public math::MathStdLib,
        public BlockSharedMemDynAlignedAlloc,
        public BlockSharedMemStMasterSync,
        public BlockSyncBarrierThread<TIdx>,
        public IntrinsicCpu,
        public rand::RandStdLib,
        public TimeStdLib,
        public warp::WarpSingleThread,
        public concepts::Implements<ConceptAcc, AccCpuThreads<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        // Partial specialization with the correct TDim and TIdx is not allowed.
        template<typename TDim2, typename TIdx2, typename TKernelFnObj, typename... TArgs>
        friend class ::alpaka::TaskKernelCpuThreads;

    private:
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST AccCpuThreads(TWorkDiv const& workDiv, std::size_t const& blockSharedMemDynSizeBytes)
            : WorkDivMembers<TDim, TIdx>(workDiv)
            , gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx)
            , bt::IdxBtRefThreadIdMap<TDim, TIdx>(m_threadToIndexMap)
            , AtomicHierarchy<
                  AtomicStdLibLock<16>, // atomics between grids
                  AtomicStdLibLock<16>, // atomics between blocks
                  AtomicStdLibLock<16> // atomics between threads
                  >()
            , math::MathStdLib()
            , BlockSharedMemDynAlignedAlloc(blockSharedMemDynSizeBytes)
            , BlockSharedMemStMasterSync(
                  [this]() { syncBlockThreads(*this); },
                  [this]() { return (m_idMasterThread == std::this_thread::get_id()); })
            , BlockSyncBarrierThread<TIdx>(getWorkDiv<Block, Threads>(workDiv).prod())
            , rand::RandStdLib()
            , TimeStdLib()
            , m_gridBlockIdx(Vec<TDim, TIdx>::zeros())
        {
        }

    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST AccCpuThreads(AccCpuThreads const&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST AccCpuThreads(AccCpuThreads&&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(AccCpuThreads const&) -> AccCpuThreads& = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(AccCpuThreads&&) -> AccCpuThreads& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~AccCpuThreads() = default;

    private:
        // getIdx
        std::mutex mutable m_mtxMapInsert; //!< The mutex used to secure insertion into the ThreadIdToIdxMap.
        typename bt::IdxBtRefThreadIdMap<TDim, TIdx>::
            ThreadIdToIdxMap mutable m_threadToIndexMap; //!< The mapping of thread id's to indices.
        Vec<TDim, TIdx> mutable m_gridBlockIdx; //!< The index of the currently executed block.

        // allocBlockSharedArr
        std::thread::id mutable m_idMasterThread; //!< The id of the master thread.
    };

    namespace traits
    {
        //#############################################################################
        //! The CPU threads accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccCpuThreads<TDim, TIdx>>
        {
            using type = AccCpuThreads<TDim, TIdx>;
        };
        //#############################################################################
        //! The CPU threads accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccCpuThreads<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccDevProps(DevCpu const& dev) -> AccDevProps<TDim, TIdx>
            {
#    ifdef ALPAKA_CI
                auto const blockThreadCountMax(static_cast<TIdx>(8));
#    else
                // \TODO: Magic number. What is the maximum? Just set a reasonable value? There is a implementation
                // defined maximum where the creation of a new thread crashes. std::thread::hardware_concurrency can
                // return 0, so 1 is the default case?
                auto const blockThreadCountMax(std::max(
                    static_cast<TIdx>(1),
                    alpaka::core::clipCast<TIdx>(std::thread::hardware_concurrency() * 8)));
#    endif
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
                        getMemBytes(dev)};
            }
        };
        //#############################################################################
        //! The CPU threads accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccCpuThreads<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccCpuThreads<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The CPU threads accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccCpuThreads<TDim, TIdx>>
        {
            using type = DevCpu;
        };

        //#############################################################################
        //! The CPU threads accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccCpuThreads<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The CPU threads accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccCpuThreads<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            //-----------------------------------------------------------------------------
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

        //#############################################################################
        //! The CPU threads execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PltfType<AccCpuThreads<TDim, TIdx>>
        {
            using type = PltfCpu;
        };

        //#############################################################################
        //! The CPU threads accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccCpuThreads<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#endif
