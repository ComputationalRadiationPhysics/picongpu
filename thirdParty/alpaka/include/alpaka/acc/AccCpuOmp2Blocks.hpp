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

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#if _OPENMP < 200203
    #error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/idx/gb/IdxGbRef.hpp>
#include <alpaka/idx/bt/IdxBtZero.hpp>
#include <alpaka/atomic/AtomicNoOp.hpp>
#include <alpaka/atomic/AtomicStlLock.hpp>
#include <alpaka/atomic/AtomicOmpCritSec.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathStl.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynBoostAlignedAlloc.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStNoSync.hpp>
#include <alpaka/block/sync/BlockSyncNoOp.hpp>
#include <alpaka/rand/RandStl.hpp>
#include <alpaka/time/TimeOmp.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/size/Traits.hpp>

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>

#include <alpaka/core/OpenMp.hpp>

#include <boost/core/ignore_unused.hpp>

#include <limits>
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
        class ExecCpuOmp2Blocks;
    }
    namespace acc
    {
        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator.
        //!
        //! This accelerator allows parallel kernel execution on a CPU device.
        //! It uses OpenMP 2.0 to implement the grid block parallelism.
        //! The block size is restricted to 1x1x1.
        template<
            typename TDim,
            typename TSize>
        class AccCpuOmp2Blocks final :
            public workdiv::WorkDivMembers<TDim, TSize>,
            public idx::gb::IdxGbRef<TDim, TSize>,
            public idx::bt::IdxBtZero<TDim, TSize>,
            public atomic::AtomicHierarchy<
                atomic::AtomicStlLock<16>,   // grid atomics
                atomic::AtomicOmpCritSec,    // block atomics
                atomic::AtomicNoOp           // thread atomics
            >,
            public math::MathStl,
            public block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc,
            public block::shared::st::BlockSharedMemStNoSync,
            public block::sync::BlockSyncNoOp,
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
            friend class ::alpaka::exec::ExecCpuOmp2Blocks;

        private:
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Blocks(
                TWorkDiv const & workDiv,
                TSize const & blockSharedMemDynSizeBytes) :
                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                    idx::gb::IdxGbRef<TDim, TSize>(m_gridBlockIdx),
                    idx::bt::IdxBtZero<TDim, TSize>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicStlLock<16>,// atomics between grids
                        atomic::AtomicOmpCritSec, // atomics between blocks
                        atomic::AtomicNoOp        // atomics between threads
                    >(),
                    math::MathStl(),
                    block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc(static_cast<std::size_t>(blockSharedMemDynSizeBytes)),
                    block::shared::st::BlockSharedMemStNoSync(),
                    block::sync::BlockSyncNoOp(),
                    rand::RandStl(),
                    time::TimeOmp(),
                    m_gridBlockIdx(vec::Vec<TDim, TSize>::zeros())
            {}

        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Blocks(AccCpuOmp2Blocks const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Blocks(AccCpuOmp2Blocks &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp2Blocks const &) -> AccCpuOmp2Blocks & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp2Blocks &&) -> AccCpuOmp2Blocks & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AccCpuOmp2Blocks() = default;

        private:
            // getIdx
            vec::Vec<TDim, TSize> mutable m_gridBlockIdx;   //!< The index of the currently executed block.
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::AccCpuOmp2Blocks<TDim, TSize>>
            {
                using type = acc::AccCpuOmp2Blocks<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::AccCpuOmp2Blocks<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

                    return {
                        // m_multiProcessorCount
                        static_cast<TSize>(1),
                        // m_gridBlockExtentMax
                        vec::Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TSize>::max(),
                        // m_blockThreadExtentMax
                        vec::Vec<TDim, TSize>::ones(),
                        // m_blockThreadCountMax
                        static_cast<TSize>(1),
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TSize>::max()};
                }
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator name trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::AccCpuOmp2Blocks<TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp2Blocks<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block accelerator device type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::AccCpuOmp2Blocks<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 block accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::AccCpuOmp2Blocks<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 block accelerator executor type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                acc::AccCpuOmp2Blocks<TDim, TSize>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuOmp2Blocks<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block executor platform type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct PltfType<
                acc::AccCpuOmp2Blocks<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 block accelerator size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::AccCpuOmp2Blocks<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
