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

// Base classes.
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/idx/gb/IdxGbRef.hpp>
#include <alpaka/idx/bt/IdxBtRefFiberIdMap.hpp>
#include <alpaka/atomic/AtomicNoOp.hpp>
#include <alpaka/atomic/AtomicStlLock.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathStl.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynBoostAlignedAlloc.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStMasterSync.hpp>
#include <alpaka/block/sync/BlockSyncBarrierFiber.hpp>
#include <alpaka/rand/RandStl.hpp>
#include <alpaka/time/TimeStl.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/size/Traits.hpp>

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>

#include <alpaka/core/Fibers.hpp>

#include <boost/core/ignore_unused.hpp>
#include <boost/predef.h>

#include <memory>
#include <typeinfo>

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuFibers;
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
            typename TSize>
        class AccCpuFibers final :
            public workdiv::WorkDivMembers<TDim, TSize>,
            public idx::gb::IdxGbRef<TDim, TSize>,
            public idx::bt::IdxBtRefFiberIdMap<TDim, TSize>,
            public atomic::AtomicHierarchy<
                atomic::AtomicStlLock<16>, // grid atomics
                atomic::AtomicStlLock<16>, // block atomics
                atomic::AtomicNoOp         // thread atomics
            >,
            public math::MathStl,
            public block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc,
            public block::shared::st::BlockSharedMemStMasterSync,
            public block::sync::BlockSyncBarrierFiber<TSize>,
            public rand::RandStl,
            public time::TimeStl
        {
        public:
            // Partial specialization with the correct TDim and TSize is not allowed.
            template<
                typename TDim2,
                typename TSize2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::exec::ExecCpuFibers;

        private:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_ACC_NO_CUDA AccCpuFibers(
                TWorkDiv const & workDiv,
                TSize const & blockSharedMemDynSizeBytes) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                    idx::gb::IdxGbRef<TDim, TSize>(m_gridBlockIdx),
                    idx::bt::IdxBtRefFiberIdMap<TDim, TSize>(m_fibersToIndices),
                    atomic::AtomicHierarchy<
                        atomic::AtomicStlLock<16>, // atomics between grids
                        atomic::AtomicStlLock<16>, // atomics between blocks
                        atomic::AtomicNoOp         // atomics between threads
                    >(),
                    math::MathStl(),
                    block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc(static_cast<std::size_t>(blockSharedMemDynSizeBytes)),
                    block::shared::st::BlockSharedMemStMasterSync(
                        [this](){block::sync::syncBlockThreads(*this);},
                        [this](){return (m_masterFiberId == boost::this_fiber::get_id());}),
                    block::sync::BlockSyncBarrierFiber<TSize>(
                        workdiv::getWorkDiv<Block, Threads>(workDiv).prod()),
                    rand::RandStl(),
                    time::TimeStl(),
                    m_gridBlockIdx(vec::Vec<TDim, TSize>::zeros())
            {}

        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuFibers(AccCpuFibers const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuFibers(AccCpuFibers &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuFibers const &) -> AccCpuFibers & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuFibers &&) -> AccCpuFibers & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AccCpuFibers() = default;

        private:
            // getIdx
            typename idx::bt::IdxBtRefFiberIdMap<TDim, TSize>::FiberIdToIdxMap mutable m_fibersToIndices;  //!< The mapping of fibers id's to indices.
            vec::Vec<TDim, TSize> mutable m_gridBlockIdx;                    //!< The index of the currently executed block.

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
                typename TSize>
            struct AccType<
                acc::AccCpuFibers<TDim, TSize>>
            {
                using type = acc::AccCpuFibers<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU fibers accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccCpuFibers<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

#ifdef ALPAKA_CI
                    auto const blockThreadCountMax(static_cast<TSize>(3));
#else
                    auto const blockThreadCountMax(static_cast<TSize>(4));  // \TODO: What is the maximum? Just set a reasonable value?
#endif
                    return {
                        // m_multiProcessorCount
                        std::max(static_cast<TSize>(1), static_cast<TSize>(std::thread::hardware_concurrency())),   // \TODO: This may be inaccurate.
                        // m_gridBlockExtentMax
                        vec::Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TSize>::max(),
                        // m_blockThreadExtentMax
                        vec::Vec<TDim, TSize>::all(blockThreadCountMax),
                        // m_blockThreadCountMax
                        blockThreadCountMax,
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TSize>::max()};
                }
            };
            //#############################################################################
            //! The CPU fibers accelerator name trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::AccCpuFibers<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccCpuFibers<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
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
                typename TSize>
            struct DevType<
                acc::AccCpuFibers<TDim, TSize>>
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
                typename TSize>
            struct DimType<
                acc::AccCpuFibers<TDim, TSize>>
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
            //! The CPU fibers accelerator executor type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccCpuFibers<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuFibers<TDim, TSize, TKernelFnObj, TArgs...>;
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
                typename TSize>
            struct PltfType<
                acc::AccCpuFibers<TDim, TSize>>
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
            //! The CPU fibers accelerator size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::AccCpuFibers<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
