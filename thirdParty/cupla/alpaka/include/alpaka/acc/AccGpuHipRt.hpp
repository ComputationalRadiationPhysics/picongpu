/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivHipBuiltIn.hpp>
#include <alpaka/idx/gb/IdxGbHipBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtHipBuiltIn.hpp>
#include <alpaka/atomic/AtomicHipBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathHipBuiltIn.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynHipBuiltIn.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStHipBuiltIn.hpp>
#include <alpaka/block/sync/BlockSyncHipBuiltIn.hpp>
#include <alpaka/rand/RandHipRand.hpp>
#include <alpaka/time/TimeHipBuiltIn.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/dev/DevHipRt.hpp>
#include <alpaka/core/Hip.hpp>

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
        class TaskKernelGpuHipRt;
    }
    namespace acc
    {
        //#############################################################################
        //! The GPU HIP accelerator.
        //!
        //! This accelerator allows parallel kernel execution on devices supporting HIP or HCC
        template<
            typename TDim,
            typename TIdx>
        class AccGpuHipRt final :
            public workdiv::WorkDivHipBuiltIn<TDim, TIdx>,
            public idx::gb::IdxGbHipBuiltIn<TDim, TIdx>,
            public idx::bt::IdxBtHipBuiltIn<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicHipBuiltIn, // grid atomics
                atomic::AtomicHipBuiltIn, // block atomics
                atomic::AtomicHipBuiltIn  // thread atomics
            >,
            public math::MathHipBuiltIn,
            public block::shared::dyn::BlockSharedMemDynHipBuiltIn,
            public block::shared::st::BlockSharedMemStHipBuiltIn,
            public block::sync::BlockSyncHipBuiltIn,
            public rand::RandHipRand,
            public time::TimeHipBuiltIn
        {
        public:
            //-----------------------------------------------------------------------------
            __device__ AccGpuHipRt(
                vec::Vec<TDim, TIdx> const & threadElemExtent) :
                    workdiv::WorkDivHipBuiltIn<TDim, TIdx>(threadElemExtent),
                    idx::gb::IdxGbHipBuiltIn<TDim, TIdx>(),
                    idx::bt::IdxBtHipBuiltIn<TDim, TIdx>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicHipBuiltIn, // atomics between grids
                        atomic::AtomicHipBuiltIn, // atomics between blocks
                        atomic::AtomicHipBuiltIn  // atomics between threads
                    >(),
                    math::MathHipBuiltIn(),
                    block::shared::dyn::BlockSharedMemDynHipBuiltIn(),
                    block::shared::st::BlockSharedMemStHipBuiltIn(),
                    block::sync::BlockSyncHipBuiltIn(),
                    rand::RandHipRand(),
                    time::TimeHipBuiltIn()
            {}

        public:
            //-----------------------------------------------------------------------------
            __device__ AccGpuHipRt(AccGpuHipRt const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ AccGpuHipRt(AccGpuHipRt &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuHipRt const &) -> AccGpuHipRt & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuHipRt &&) -> AccGpuHipRt & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC ~AccGpuHipRt() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccGpuHipRt<TDim, TIdx>>
            {
                using type = acc::AccGpuHipRt<TDim, TIdx>;
            };
            //#############################################################################
            //! The GPU HIP accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccGpuHipRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevHipRt const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
                    hipDeviceProp_t hipDevProp;
                    ALPAKA_HIP_RT_CHECK(hipGetDeviceProperties(
                        &hipDevProp,
                        dev.m_iDevice));

                    return {
                        // m_multiProcessorCount
                        alpaka::core::clipCast<TIdx>(hipDevProp.multiProcessorCount),
                        // m_gridBlockExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxGridSize[2u]),
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxGridSize[1u]),
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxGridSize[0u]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxThreadsDim[2u]),
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxThreadsDim[1u]),
                                alpaka::core::clipCast<TIdx>(hipDevProp.maxThreadsDim[0u]))),
                        // m_blockThreadCountMax
                        alpaka::core::clipCast<TIdx>(hipDevProp.maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max()};
                }
            };
            //#############################################################################
            //! The GPU Hip accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccGpuHipRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccGpuHipRt<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator device type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccGpuHipRt<TDim, TIdx>>
            {
                using type = dev::DevHipRt;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccGpuHipRt<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace kernel
    {
        namespace detail
        {
            //#############################################################################
            //! specialization of the TKernelFnObj return type evaluation
            //
            // It is not possible to determine the result type of a __device__ lambda for CUDA on the host side.
            // https://github.com/ComputationalRadiationPhysics/alpaka/pull/695#issuecomment-446103194
            // The execution task TaskKernelGpuHipRt is therefore performing this check on device side.
            template<
                typename TDim,
                typename TIdx>
            struct CheckFnReturnType<
                acc::AccGpuHipRt<
                    TDim,
                    TIdx>>
            {
                template<
                    typename TKernelFnObj,
                    typename... TArgs>
                void operator()(
                    TKernelFnObj const &,
                    TArgs const & ...)
                {

                }
            };
        }
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator execution task type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskKernel<
                acc::AccGpuHipRt<TDim, TIdx>,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto createTaskKernel(
                    TWorkDiv const & workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs const & ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> kernel::TaskKernelGpuHipRt<
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    TArgs...>
#endif
                {
                    return
                        kernel::TaskKernelGpuHipRt<
                            TDim,
                            TIdx,
                            TKernelFnObj,
                            TArgs...>(
                                workDiv,
                                kernelFnObj,
                                args...);
                }
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU HIP execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccGpuHipRt<TDim, TIdx>>
            {
                using type = pltf::PltfHipRt;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccGpuHipRt<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
