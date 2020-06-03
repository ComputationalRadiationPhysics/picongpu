/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivCudaBuiltIn.hpp>
#include <alpaka/idx/gb/IdxGbCudaBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtCudaBuiltIn.hpp>
#include <alpaka/atomic/AtomicCudaBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathCudaBuiltIn.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynCudaBuiltIn.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStCudaBuiltIn.hpp>
#include <alpaka/block/sync/BlockSyncCudaBuiltIn.hpp>
#include <alpaka/rand/RandCuRand.hpp>
#include <alpaka/time/TimeCudaBuiltIn.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Cuda.hpp>
#include <alpaka/dev/DevCudaRt.hpp>

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
        class TaskKernelGpuCudaRt;
    }
    namespace acc
    {
        //#############################################################################
        //! The GPU CUDA accelerator.
        //!
        //! This accelerator allows parallel kernel execution on devices supporting CUDA.
        template<
            typename TDim,
            typename TIdx>
        class AccGpuCudaRt final :
            public workdiv::WorkDivCudaBuiltIn<TDim, TIdx>,
            public idx::gb::IdxGbCudaBuiltIn<TDim, TIdx>,
            public idx::bt::IdxBtCudaBuiltIn<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicCudaBuiltIn, // grid atomics
                atomic::AtomicCudaBuiltIn, // block atomics
                atomic::AtomicCudaBuiltIn  // thread atomics
            >,
            public math::MathCudaBuiltIn,
            public block::shared::dyn::BlockSharedMemDynCudaBuiltIn,
            public block::shared::st::BlockSharedMemStCudaBuiltIn,
            public block::sync::BlockSyncCudaBuiltIn,
            public rand::RandCuRand,
            public time::TimeCudaBuiltIn,
            public concepts::Implements<ConceptAcc, AccGpuCudaRt<TDim, TIdx>>
        {
        public:
            //-----------------------------------------------------------------------------
            __device__ AccGpuCudaRt(
                vec::Vec<TDim, TIdx> const & threadElemExtent) :
                    workdiv::WorkDivCudaBuiltIn<TDim, TIdx>(threadElemExtent),
                    idx::gb::IdxGbCudaBuiltIn<TDim, TIdx>(),
                    idx::bt::IdxBtCudaBuiltIn<TDim, TIdx>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicCudaBuiltIn, // atomics between grids
                        atomic::AtomicCudaBuiltIn, // atomics between blocks
                        atomic::AtomicCudaBuiltIn  // atomics between threads
                    >(),
                    math::MathCudaBuiltIn(),
                    block::shared::dyn::BlockSharedMemDynCudaBuiltIn(),
                    block::shared::st::BlockSharedMemStCudaBuiltIn(),
                    block::sync::BlockSyncCudaBuiltIn(),
                    rand::RandCuRand(),
                    time::TimeCudaBuiltIn()
            {}

        public:
            //-----------------------------------------------------------------------------
            __device__ AccGpuCudaRt(AccGpuCudaRt const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ AccGpuCudaRt(AccGpuCudaRt &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuCudaRt const &) -> AccGpuCudaRt & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuCudaRt &&) -> AccGpuCudaRt & = delete;
            //-----------------------------------------------------------------------------
            ~AccGpuCudaRt() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccGpuCudaRt<TDim, TIdx>>
            {
                using type = acc::AccGpuCudaRt<TDim, TIdx>;
            };
            //#############################################################################
            //! The GPU CUDA accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccGpuCudaRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCudaRt const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
                    // Reading only the necessary attributes with cudaDeviceGetAttribute is faster than reading all with cudaGetDeviceProperties
                    // https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/
                    int multiProcessorCount = {};
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &multiProcessorCount,
                        cudaDevAttrMultiProcessorCount,
                        dev.m_iDevice));

                    int maxGridSize[3] = {};
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxGridSize[0],
                        cudaDevAttrMaxGridDimX,
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxGridSize[1],
                        cudaDevAttrMaxGridDimY,
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxGridSize[2],
                        cudaDevAttrMaxGridDimZ,
                        dev.m_iDevice));

                    int maxBlockDim[3] = {};
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxBlockDim[0],
                        cudaDevAttrMaxBlockDimX,
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxBlockDim[1],
                        cudaDevAttrMaxBlockDimY,
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxBlockDim[2],
                        cudaDevAttrMaxBlockDimZ,
                        dev.m_iDevice));

                    int maxThreadsPerBlock = {};
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceGetAttribute(
                        &maxThreadsPerBlock,
                        cudaDevAttrMaxThreadsPerBlock,
                        dev.m_iDevice));

                    return {
                        // m_multiProcessorCount
                        alpaka::core::clipCast<TIdx>(multiProcessorCount),
                        // m_gridBlockExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                alpaka::core::clipCast<TIdx>(maxGridSize[2u]),
                                alpaka::core::clipCast<TIdx>(maxGridSize[1u]),
                                alpaka::core::clipCast<TIdx>(maxGridSize[0u]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                alpaka::core::clipCast<TIdx>(maxBlockDim[2u]),
                                alpaka::core::clipCast<TIdx>(maxBlockDim[1u]),
                                alpaka::core::clipCast<TIdx>(maxBlockDim[0u]))),
                        // m_blockThreadCountMax
                        alpaka::core::clipCast<TIdx>(maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max()};
                }
            };
            //#############################################################################
            //! The GPU CUDA accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccGpuCudaRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccGpuCudaRt<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator device type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccGpuCudaRt<TDim, TIdx>>
            {
                using type = dev::DevCudaRt;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccGpuCudaRt<TDim, TIdx>>
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
            // The execution task TaskKernelGpuCudaRt is therefore performing this check on device side.
            template<
                typename TDim,
                typename TIdx>
            struct CheckFnReturnType<
                acc::AccGpuCudaRt<
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
            //! The GPU CUDA accelerator execution task type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskKernel<
                acc::AccGpuCudaRt<TDim, TIdx>,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto createTaskKernel(
                    TWorkDiv const & workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs && ... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> kernel::TaskKernelGpuCudaRt<
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    TArgs...>
#endif
                {
                    return
                        kernel::TaskKernelGpuCudaRt<
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
            //! The CPU CUDA execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccGpuCudaRt<TDim, TIdx>>
            {
                using type = pltf::PltfCudaRt;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccGpuCudaRt<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
