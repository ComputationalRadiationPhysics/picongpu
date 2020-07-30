/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

// Base classes.
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/idx/gb/IdxGbRef.hpp>
#include <alpaka/idx/bt/IdxBtRefFiberIdMap.hpp>
#include <alpaka/atomic/AtomicNoOp.hpp>
#include <alpaka/atomic/AtomicStdLibLock.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathStdLib.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynAlignedAlloc.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStMasterSync.hpp>
#include <alpaka/block/sync/BlockSyncBarrierFiber.hpp>
#include <alpaka/intrinsic/IntrinsicCpu.hpp>
#include <alpaka/rand/RandStdLib.hpp>
#include <alpaka/time/TimeStdLib.hpp>
#include <alpaka/warp/WarpSingleThread.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Fibers.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>

#include <memory>
#include <thread>
#include <typeinfo>

namespace alpaka
{
    namespace kernel
    {
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelCpuFibers;
    }
    namespace acc
    {
        //#############################################################################
        //! The CPU fibers accelerator.
        //!
        //! This accelerator allows parallel kernel execution on a CPU device.
        //! It uses boost::fibers to implement the cooperative parallelism.
        //! By using fibers the shared memory can reside in the closest memory/cache available.
        //! Furthermore there is no false sharing between neighboring threads as it is the case in real multi-threading.
        template<
            typename TDim,
            typename TIdx>
        class AccCpuFibers final :
            public workdiv::WorkDivMembers<TDim, TIdx>,
            public idx::gb::IdxGbRef<TDim, TIdx>,
            public idx::bt::IdxBtRefFiberIdMap<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicStdLibLock<16>, // grid atomics
                atomic::AtomicStdLibLock<16>, // block atomics
                atomic::AtomicNoOp         // thread atomics
            >,
            public math::MathStdLib,
            public block::shared::dyn::BlockSharedMemDynAlignedAlloc,
            public block::shared::st::BlockSharedMemStMasterSync,
            public block::sync::BlockSyncBarrierFiber<TIdx>,
            public intrinsic::IntrinsicCpu,
            public rand::RandStdLib,
            public time::TimeStdLib,
            public warp::WarpSingleThread,
            public concepts::Implements<ConceptAcc, AccCpuFibers<TDim, TIdx>>
        {
        public:
            // Partial specialization with the correct TDim and TIdx is not allowed.
            template<
                typename TDim2,
                typename TIdx2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::kernel::TaskKernelCpuFibers;

        private:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST AccCpuFibers(
                TWorkDiv const & workDiv,
                TIdx const & blockSharedMemDynSizeBytes) :
                    workdiv::WorkDivMembers<TDim, TIdx>(workDiv),
                    idx::gb::IdxGbRef<TDim, TIdx>(m_gridBlockIdx),
                    idx::bt::IdxBtRefFiberIdMap<TDim, TIdx>(m_fibersToIndices),
                    atomic::AtomicHierarchy<
                        atomic::AtomicStdLibLock<16>, // atomics between grids
                        atomic::AtomicStdLibLock<16>, // atomics between blocks
                        atomic::AtomicNoOp         // atomics between threads
                    >(),
                    math::MathStdLib(),
                    block::shared::dyn::BlockSharedMemDynAlignedAlloc(static_cast<std::size_t>(blockSharedMemDynSizeBytes)),
                    block::shared::st::BlockSharedMemStMasterSync(
                        [this](){block::sync::syncBlockThreads(*this);},
                        [this](){return (m_masterFiberId == boost::this_fiber::get_id());}),
                    block::sync::BlockSyncBarrierFiber<TIdx>(
                        workdiv::getWorkDiv<Block, Threads>(workDiv).prod()),
                    rand::RandStdLib(),
                    time::TimeStdLib(),
                    m_gridBlockIdx(vec::Vec<TDim, TIdx>::zeros())
            {}

        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST AccCpuFibers(AccCpuFibers const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST AccCpuFibers(AccCpuFibers &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(AccCpuFibers const &) -> AccCpuFibers & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(AccCpuFibers &&) -> AccCpuFibers & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AccCpuFibers() = default;

        private:
            // getIdx
            typename idx::bt::IdxBtRefFiberIdMap<TDim, TIdx>::FiberIdToIdxMap mutable m_fibersToIndices;  //!< The mapping of fibers id's to indices.
            vec::Vec<TDim, TIdx> mutable m_gridBlockIdx;                    //!< The index of the currently executed block.

            // allocBlockSharedArr
            boost::fibers::fiber::id mutable m_masterFiberId;           //!< The id of the master fiber.
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccCpuFibers<TDim, TIdx>>
            {
                using type = acc::AccCpuFibers<TDim, TIdx>;
            };
            //#############################################################################
            //! The CPU fibers accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccCpuFibers<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim, TIdx>
                {
#ifdef ALPAKA_CI
                    auto const blockThreadCountMax(static_cast<TIdx>(3));
#else
                    auto const blockThreadCountMax(static_cast<TIdx>(4));  // \TODO: What is the maximum? Just set a reasonable value?
#endif
                    return {
                        // m_multiProcessorCount
                        std::max(static_cast<TIdx>(1), alpaka::core::clipCast<TIdx>(std::thread::hardware_concurrency())),   // \TODO: This may be inaccurate.
                        // m_gridBlockExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        vec::Vec<TDim, TIdx>::all(blockThreadCountMax),
                        // m_blockThreadCountMax
                        blockThreadCountMax,
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        dev::getMemBytes( dev )};
                }
            };
            //#############################################################################
            //! The CPU fibers accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccCpuFibers<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccCpuFibers<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator device type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccCpuFibers<TDim, TIdx>>
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
            //! The CPU fibers accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccCpuFibers<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator execution task type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskKernel<
                acc::AccCpuFibers<TDim, TIdx>,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto createTaskKernel(
                    TWorkDiv const & workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs && ... args)
                {
                    return
                        kernel::TaskKernelCpuFibers<
                            TDim,
                            TIdx,
                            TKernelFnObj,
                            TArgs...>(
                                workDiv,
                                kernelFnObj,
                                std::forward<TArgs>(args)...);
                }
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccCpuFibers<TDim, TIdx>>
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
            //! The CPU fibers accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccCpuFibers<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
