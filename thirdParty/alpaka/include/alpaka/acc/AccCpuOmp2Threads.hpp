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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED

#if _OPENMP < 200203
    #error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers
#include <alpaka/idx/gb/IdxGbRef.hpp>           // IdxGbRef
#include <alpaka/idx/bt/IdxBtOmp.hpp>           // IdxBtOmp
#include <alpaka/atomic/AtomicStlLock.hpp>      // AtomicStlLock
#include <alpaka/atomic/AtomicOmpCritSec.hpp>   // AtomicOmpCritSec
#include <alpaka/atomic/AtomicHierarchy.hpp>    // AtomicHierarchy
#include <alpaka/math/MathStl.hpp>              // MathStl
#include <alpaka/block/shared/dyn/BlockSharedMemDynBoostAlignedAlloc.hpp>   // BlockSharedMemDynBoostAlignedAlloc
#include <alpaka/block/shared/st/BlockSharedMemStMasterSync.hpp>            // BlockSharedMemStMasterSync
#include <alpaka/block/sync/BlockSyncBarrierOmp.hpp>                        // BlockSyncBarrierOmp
#include <alpaka/rand/RandStl.hpp>              // RandStl
#include <alpaka/time/TimeOmp.hpp>              // TimeOmp

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // acc::traits::AccType
#include <alpaka/exec/Traits.hpp>               // exec::traits::ExecType
#include <alpaka/dev/Traits.hpp>                // dev::traits::DevType
#include <alpaka/pltf/Traits.hpp>               // pltf::traits::PltfType
#include <alpaka/size/Traits.hpp>               // size::traits::SizeType

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                // dev::DevCpu

#include <alpaka/core/OpenMp.hpp>

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused

#include <limits>                               // std::numeric_limits
#include <typeinfo>                             // typeid

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuOmp2Threads;
    }
    namespace acc
    {
        //#############################################################################
        //! The CPU OpenMP 2.0 thread accelerator.
        //!
        //! This accelerator allows parallel kernel execution on a CPU device.
        //! It uses OpenMP 2.0 to implement the block thread parallelism.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class AccCpuOmp2Threads final :
            public workdiv::WorkDivMembers<TDim, TSize>,
            public idx::gb::IdxGbRef<TDim, TSize>,
            public idx::bt::IdxBtOmp<TDim, TSize>,
            public atomic::AtomicHierarchy<
                atomic::AtomicStlLock,       // grid atomics
                atomic::AtomicOmpCritSec,    // block atomics
                atomic::AtomicOmpCritSec     // thread atomics
            >,
            public math::MathStl,
            public block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc,
            public block::shared::st::BlockSharedMemStMasterSync,
            public block::sync::BlockSyncBarrierOmp,
            public rand::RandStl,
            public time::TimeOmp
        {
        public:
            // Partial specialization with the correct TDim and TSize is not allowed.
            template<
                typename TDim2,
                typename TSize2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::exec::ExecCpuOmp2Threads;

        private:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Threads(
                TWorkDiv const & workDiv,
                TSize const & blockSharedMemDynSizeBytes) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                    idx::gb::IdxGbRef<TDim, TSize>(m_gridBlockIdx),
                    idx::bt::IdxBtOmp<TDim, TSize>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicStlLock,    // atomics between grids
                        atomic::AtomicOmpCritSec, // atomics between blocks
                        atomic::AtomicOmpCritSec  // atomics between threads
                    >(),
                    math::MathStl(),
                    block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc(static_cast<std::size_t>(blockSharedMemDynSizeBytes)),
                    block::shared::st::BlockSharedMemStMasterSync(
                        [this](){block::sync::syncBlockThreads(*this);},
                        [](){return (::omp_get_thread_num() == 0);}),
                    block::sync::BlockSyncBarrierOmp(),
                    rand::RandStl(),
                    time::TimeOmp(),
                    m_gridBlockIdx(vec::Vec<TDim, TSize>::zeros())
            {}

        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Threads(AccCpuOmp2Threads const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Threads(AccCpuOmp2Threads &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp2Threads const &) -> AccCpuOmp2Threads & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp2Threads &&) -> AccCpuOmp2Threads & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~AccCpuOmp2Threads() = default;

        private:
            // getIdx
            vec::Vec<TDim, TSize> mutable m_gridBlockIdx;  //!< The index of the currently executed block.
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::AccCpuOmp2Threads<TDim, TSize>>
            {
                using type = acc::AccCpuOmp2Threads<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccCpuOmp2Threads<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

                    // m_blockThreadCountMax
#ifdef ALPAKA_CI
                    auto const blockThreadCountMax(static_cast<TSize>(4));
#else
                    auto const blockThreadCountMax(static_cast<TSize>(omp::getMaxOmpThreads()));
#endif
                    return {
                        // m_multiProcessorCount
                        static_cast<TSize>(1),
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
            //! The CPU OpenMP 2.0 thread accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::AccCpuOmp2Threads<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp2Threads<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::AccCpuOmp2Threads<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 thread accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::AccCpuOmp2Threads<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 thread accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccCpuOmp2Threads<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuOmp2Threads<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread executor platform type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct PltfType<
                acc::AccCpuOmp2Threads<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 thread accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::AccCpuOmp2Threads<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
