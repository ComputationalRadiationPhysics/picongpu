/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivUniformCudaHipBuiltIn.hpp>
#include <alpaka/idx/gb/IdxGbUniformCudaHipBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtUniformCudaHipBuiltIn.hpp>
#include <alpaka/atomic/AtomicUniformCudaHipBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathUniformCudaHipBuiltIn.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynUniformCudaHipBuiltIn.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStUniformCudaHipBuiltIn.hpp>
#include <alpaka/block/sync/BlockSyncUniformCudaHipBuiltIn.hpp>
#include <alpaka/intrinsic/IntrinsicUniformCudaHipBuiltIn.hpp>
#include <alpaka/rand/RandUniformCudaHipRand.hpp>
#include <alpaka/time/TimeUniformCudaHipBuiltIn.hpp>
#include <alpaka/warp/WarpUniformCudaHipBuiltIn.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Cuda.hpp>
#include <alpaka/dev/DevUniformCudaHipRt.hpp>

#include <typeinfo>

namespace alpaka
{
    namespace kernel
    {
        template<
            typename TAcc,
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelGpuUniformCudaHipRt;
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
        class AccGpuUniformCudaHipRt :
            public workdiv::WorkDivUniformCudaHipBuiltIn<TDim, TIdx>,
            public idx::gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>,
            public idx::bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicUniformCudaHipBuiltIn, // grid atomics
                atomic::AtomicUniformCudaHipBuiltIn, // block atomics
                atomic::AtomicUniformCudaHipBuiltIn  // thread atomics
            >,
            public math::MathUniformCudaHipBuiltIn,
            public block::shared::dyn::BlockSharedMemDynUniformCudaHipBuiltIn,
            public block::shared::st::BlockSharedMemStUniformCudaHipBuiltIn,
            public block::sync::BlockSyncUniformCudaHipBuiltIn,
            public intrinsic::IntrinsicUniformCudaHipBuiltIn,
            public rand::RandUniformCudaHipRand,
            public time::TimeUniformCudaHipBuiltIn,
            public warp::WarpUniformCudaHipBuiltIn,
            public concepts::Implements<ConceptAcc, AccGpuUniformCudaHipRt<TDim, TIdx>>
        {
        public:
            //-----------------------------------------------------------------------------
            __device__ AccGpuUniformCudaHipRt(
                vec::Vec<TDim, TIdx> const & threadElemExtent) :
                    workdiv::WorkDivUniformCudaHipBuiltIn<TDim, TIdx>(threadElemExtent),
                    idx::gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>(),
                    idx::bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicUniformCudaHipBuiltIn, // atomics between grids
                        atomic::AtomicUniformCudaHipBuiltIn, // atomics between blocks
                        atomic::AtomicUniformCudaHipBuiltIn  // atomics between threads
                    >(),
                    math::MathUniformCudaHipBuiltIn(),
                    block::shared::dyn::BlockSharedMemDynUniformCudaHipBuiltIn(),
                    block::shared::st::BlockSharedMemStUniformCudaHipBuiltIn(),
                    block::sync::BlockSyncUniformCudaHipBuiltIn(),
                    rand::RandUniformCudaHipRand(),
                    time::TimeUniformCudaHipBuiltIn()
            {}

        public:

            //using baseType = AccUniformCudaHip<TDim,TIdx>;

            //-----------------------------------------------------------------------------
            __device__ AccGpuUniformCudaHipRt(AccGpuUniformCudaHipRt const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ AccGpuUniformCudaHipRt(AccGpuUniformCudaHipRt &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuUniformCudaHipRt const &) -> AccGpuUniformCudaHipRt & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(AccGpuUniformCudaHipRt &&) -> AccGpuUniformCudaHipRt & = delete;
            //-----------------------------------------------------------------------------
            ~AccGpuUniformCudaHipRt() = default;
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
                acc::AccGpuUniformCudaHipRt<TDim, TIdx>>
            {
                using type = acc::AccGpuUniformCudaHipRt<TDim, TIdx>;
            };
            //#############################################################################
            //! The GPU CUDA accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccGpuUniformCudaHipRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevUniformCudaHipRt const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    // Reading only the necessary attributes with cudaDeviceGetAttribute is faster than reading all with cuda
                    // https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/
                    int multiProcessorCount = {};
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaDeviceGetAttribute(
                        &multiProcessorCount,
                        cudaDevAttrMultiProcessorCount,
                        dev.m_iDevice));

                    int maxGridSize[3] = {};
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaDeviceGetAttribute(
                        &maxGridSize[0],
                        cudaDevAttrMaxGridDimX,
                        dev.m_iDevice));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaDeviceGetAttribute(
                        &maxGridSize[1],
                        cudaDevAttrMaxGridDimY,
                        dev.m_iDevice));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaDeviceGetAttribute(
                        &maxGridSize[2],
                        cudaDevAttrMaxGridDimZ,
                        dev.m_iDevice));

                    int maxBlockDim[3] = {};
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaDeviceGetAttribute(
                        &maxBlockDim[0],
                        cudaDevAttrMaxBlockDimX,
                        dev.m_iDevice));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaDeviceGetAttribute(
                        &maxBlockDim[1],
                        cudaDevAttrMaxBlockDimY,
                        dev.m_iDevice));
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaDeviceGetAttribute(
                        &maxBlockDim[2],
                        cudaDevAttrMaxBlockDimZ,
                        dev.m_iDevice));

                    int maxThreadsPerBlock = {};
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaDeviceGetAttribute(
                        &maxThreadsPerBlock,
                        cudaDevAttrMaxThreadsPerBlock,
                        dev.m_iDevice));

                    int sharedMemSizeBytes = {};
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaDeviceGetAttribute(
                        &sharedMemSizeBytes,
                        cudaDevAttrMaxSharedMemoryPerBlock,
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
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(sharedMemSizeBytes)
                    };

#else
                    hipDeviceProp_t hipDevProp;
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipGetDeviceProperties(
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
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(hipDevProp.sharedMemPerBlock)
                    };
#endif
                }
            };
            //#############################################################################
            //! The GPU CUDA accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccGpuUniformCudaHipRt<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccGpuUniformCudaHipRt<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
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
                acc::AccGpuUniformCudaHipRt<TDim, TIdx>>
            {
                using type = dev::DevUniformCudaHipRt;
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
                acc::AccGpuUniformCudaHipRt<TDim, TIdx>>
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
            // https://github.com/alpaka-group/alpaka/pull/695#issuecomment-446103194
            // The execution task TaskKernelGpuUniformCudaHipRt is therefore performing this check on device side.
            template<
                typename TDim,
                typename TIdx>
            struct CheckFnReturnType<
                acc::AccGpuUniformCudaHipRt<
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
                acc::AccGpuUniformCudaHipRt<TDim, TIdx>,
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
                        kernel::TaskKernelGpuUniformCudaHipRt<
                            acc::AccGpuUniformCudaHipRt<TDim, TIdx>,
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
                acc::AccGpuUniformCudaHipRt<TDim, TIdx>>
            {
                using type = pltf::PltfUniformCudaHipRt;
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
                acc::AccGpuUniformCudaHipRt<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
