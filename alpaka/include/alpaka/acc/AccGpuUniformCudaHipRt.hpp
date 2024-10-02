/* Copyright 2022 Benjamin Worpitz, René Widera, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "alpaka/atomic/AtomicHierarchy.hpp"
#include "alpaka/atomic/AtomicUniformCudaHipBuiltIn.hpp"
#include "alpaka/block/shared/dyn/BlockSharedMemDynUniformCudaHipBuiltIn.hpp"
#include "alpaka/block/shared/st/BlockSharedMemStUniformCudaHipBuiltIn.hpp"
#include "alpaka/block/sync/BlockSyncUniformCudaHipBuiltIn.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/idx/bt/IdxBtUniformCudaHipBuiltIn.hpp"
#include "alpaka/idx/gb/IdxGbUniformCudaHipBuiltIn.hpp"
#include "alpaka/intrinsic/IntrinsicUniformCudaHipBuiltIn.hpp"
#include "alpaka/math/MathUniformCudaHipBuiltIn.hpp"
#include "alpaka/mem/fence/MemFenceUniformCudaHipBuiltIn.hpp"
#include "alpaka/rand/RandDefault.hpp"
#include "alpaka/rand/RandUniformCudaHipRand.hpp"
#include "alpaka/warp/WarpUniformCudaHipBuiltIn.hpp"
#include "alpaka/workdiv/WorkDivUniformCudaHipBuiltIn.hpp"

// Specialized traits.
#include "alpaka/acc/Traits.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/Traits.hpp"

// Implementation details.
#include "alpaka/core/ClipCast.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"

#include <typeinfo>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelGpuUniformCudaHipRt;

    //! The GPU CUDA accelerator.
    //!
    //! This accelerator allows parallel kernel execution on devices supporting CUDA.
    template<typename TApi, typename TDim, typename TIdx>
    class AccGpuUniformCudaHipRt final
        : public WorkDivUniformCudaHipBuiltIn<TDim, TIdx>
        , public gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>
        , public bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>
        , public AtomicHierarchy<
              AtomicUniformCudaHipBuiltIn, // grid atomics
              AtomicUniformCudaHipBuiltIn, // block atomics
              AtomicUniformCudaHipBuiltIn> // thread atomics
        , public math::MathUniformCudaHipBuiltIn
        , public BlockSharedMemDynUniformCudaHipBuiltIn
        , public BlockSharedMemStUniformCudaHipBuiltIn
        , public BlockSyncUniformCudaHipBuiltIn
        , public IntrinsicUniformCudaHipBuiltIn
        , public MemFenceUniformCudaHipBuiltIn
#    ifdef ALPAKA_DISABLE_VENDOR_RNG
        , public rand::RandDefault
#    else
        , public rand::RandUniformCudaHipRand<TApi>
#    endif
        , public warp::WarpUniformCudaHipBuiltIn
        , public concepts::Implements<ConceptAcc, AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        AccGpuUniformCudaHipRt(AccGpuUniformCudaHipRt const&) = delete;
        AccGpuUniformCudaHipRt(AccGpuUniformCudaHipRt&&) = delete;
        auto operator=(AccGpuUniformCudaHipRt const&) -> AccGpuUniformCudaHipRt& = delete;
        auto operator=(AccGpuUniformCudaHipRt&&) -> AccGpuUniformCudaHipRt& = delete;

        ALPAKA_FN_HOST_ACC AccGpuUniformCudaHipRt(Vec<TDim, TIdx> const& threadElemExtent)
            : WorkDivUniformCudaHipBuiltIn<TDim, TIdx>(threadElemExtent)
        {
        }
    };

    namespace trait
    {
        //! The GPU CUDA accelerator accelerator type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct AccType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = AccGpuUniformCudaHipRt<TApi, TDim, TIdx>;
        };

        //! The GPU CUDA accelerator device properties get trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct GetAccDevProps<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccDevProps(DevUniformCudaHipRt<TApi> const& dev) -> AccDevProps<TDim, TIdx>
            {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                // Reading only the necessary attributes with cudaDeviceGetAttribute is faster than reading all with
                // cuda https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/
                int multiProcessorCount = {};
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &multiProcessorCount,
                    TApi::deviceAttributeMultiprocessorCount,
                    dev.getNativeHandle()));

                int maxGridSize[3] = {};
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxGridSize[0],
                    TApi::deviceAttributeMaxGridDimX,
                    dev.getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxGridSize[1],
                    TApi::deviceAttributeMaxGridDimY,
                    dev.getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxGridSize[2],
                    TApi::deviceAttributeMaxGridDimZ,
                    dev.getNativeHandle()));

                int maxBlockDim[3] = {};
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxBlockDim[0],
                    TApi::deviceAttributeMaxBlockDimX,
                    dev.getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxBlockDim[1],
                    TApi::deviceAttributeMaxBlockDimY,
                    dev.getNativeHandle()));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxBlockDim[2],
                    TApi::deviceAttributeMaxBlockDimZ,
                    dev.getNativeHandle()));

                int maxThreadsPerBlock = {};
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &maxThreadsPerBlock,
                    TApi::deviceAttributeMaxThreadsPerBlock,
                    dev.getNativeHandle()));

                int sharedMemSizeBytes = {};
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceGetAttribute(
                    &sharedMemSizeBytes,
                    TApi::deviceAttributeMaxSharedMemoryPerBlock,
                    dev.getNativeHandle()));

                return {// m_multiProcessorCount
                        alpaka::core::clipCast<TIdx>(multiProcessorCount),
                        // m_gridBlockExtentMax
                        getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                            alpaka::core::clipCast<TIdx>(maxGridSize[2u]),
                            alpaka::core::clipCast<TIdx>(maxGridSize[1u]),
                            alpaka::core::clipCast<TIdx>(maxGridSize[0u]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                            alpaka::core::clipCast<TIdx>(maxBlockDim[2u]),
                            alpaka::core::clipCast<TIdx>(maxBlockDim[1u]),
                            alpaka::core::clipCast<TIdx>(maxBlockDim[0u]))),
                        // m_blockThreadCountMax
                        alpaka::core::clipCast<TIdx>(maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(sharedMemSizeBytes)};

#    else
                typename TApi::DeviceProp_t properties;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getDeviceProperties(&properties, dev.getNativeHandle()));

                return {// m_multiProcessorCount
                        alpaka::core::clipCast<TIdx>(properties.multiProcessorCount),
                        // m_gridBlockExtentMax
                        getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                            alpaka::core::clipCast<TIdx>(properties.maxGridSize[2u]),
                            alpaka::core::clipCast<TIdx>(properties.maxGridSize[1u]),
                            alpaka::core::clipCast<TIdx>(properties.maxGridSize[0u]))),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                            alpaka::core::clipCast<TIdx>(properties.maxThreadsDim[2u]),
                            alpaka::core::clipCast<TIdx>(properties.maxThreadsDim[1u]),
                            alpaka::core::clipCast<TIdx>(properties.maxThreadsDim[0u]))),
                        // m_blockThreadCountMax
                        alpaka::core::clipCast<TIdx>(properties.maxThreadsPerBlock),
                        // m_threadElemExtentMax
                        Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_sharedMemSizeBytes
                        static_cast<size_t>(properties.sharedMemPerBlock)};
#    endif
            }
        };

        //! The GPU CUDA accelerator name trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct GetAccName<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return std::string("AccGpu") + TApi::name + "Rt<" + std::to_string(TDim::value) + ","
                       + core::demangled<TIdx> + ">";
            }
        };

        //! The GPU CUDA accelerator device type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct DevType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = DevUniformCudaHipRt<TApi>;
        };

        //! The GPU CUDA accelerator dimension getter trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct DimType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = TDim;
        };
    } // namespace trait

    namespace detail
    {
        //! specialization of the TKernelFnObj return type evaluation
        //
        // It is not possible to determine the result type of a __device__ lambda for CUDA on the host side.
        // https://github.com/alpaka-group/alpaka/pull/695#issuecomment-446103194
        // The execution task TaskKernelGpuUniformCudaHipRt is therefore performing this check on device side.
        template<typename TApi, typename TDim, typename TIdx>
        struct CheckFnReturnType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            template<typename TKernelFnObj, typename... TArgs>
            void operator()(TKernelFnObj const&, TArgs const&...)
            {
            }
        };
    } // namespace detail

    namespace trait
    {
        //! The GPU CUDA accelerator execution task type trait specialization.
        template<
            typename TApi,
            typename TDim,
            typename TIdx,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        struct CreateTaskKernel<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelGpuUniformCudaHipRt<
                    TApi,
                    AccGpuUniformCudaHipRt<TApi, TDim, TIdx>,
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    TArgs...>(workDiv, kernelFnObj, std::forward<TArgs>(args)...);
            }
        };

        //! The CPU CUDA execution task platform type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct PlatformType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = PlatformUniformCudaHipRt<TApi>;
        };

        //! The GPU CUDA accelerator idx type trait specialization.
        template<typename TApi, typename TDim, typename TIdx>
        struct IdxType<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka

#endif
