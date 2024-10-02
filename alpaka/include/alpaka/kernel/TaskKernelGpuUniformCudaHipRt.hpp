/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Jan Stephan, Andrea Bocci, Bernhard
 * Manfred Gruber, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGpuUniformCudaHipRt.hpp"
#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Decay.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/core/RemoveRestrict.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtBlocking.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/workdiv/WorkDivHelpers.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <stdexcept>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    include <iostream>
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    if !defined(ALPAKA_HOST_ONLY)

#        include "alpaka/core/BoostPredef.hpp"

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

namespace alpaka
{
    namespace detail
    {
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wunused-template"
#        endif
        //! The GPU CUDA/HIP kernel entry point.
        // \NOTE: 'A __global__ function or function template cannot have a trailing return type.'
        // We have put the function into a shallow namespace and gave it a short name, so the mangled name in the
        // profiler (e.g. ncu) is as shorter as possible.
        template<typename TKernelFnObj, typename TApi, typename TAcc, typename TDim, typename TIdx, typename... TArgs>
        __global__ void gpuKernel(
            Vec<TDim, TIdx> const threadElemExtent,
            TKernelFnObj const kernelFnObj,
            TArgs... args)
        {
#        if BOOST_ARCH_PTX && (BOOST_ARCH_PTX < BOOST_VERSION_NUMBER(2, 0, 0))
#            error "Device capability >= 2.0 is required!"
#        endif

            const TAcc acc(threadElemExtent);

// with clang it is not possible to query std::result_of for a pure device lambda created on the host side
#        if !(BOOST_COMP_CLANG_CUDA && BOOST_COMP_CLANG)
            static_assert(
                std::is_same_v<decltype(kernelFnObj(const_cast<TAcc const&>(acc), args...)), void>,
                "The TKernelFnObj is required to return void!");
#        endif
            kernelFnObj(const_cast<TAcc const&>(acc), args...);
        }
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic pop
#        endif
    } // namespace detail

    namespace uniform_cuda_hip
    {
        namespace detail
        {
            template<typename TDim, typename TIdx>
            ALPAKA_FN_HOST auto checkVecOnly3Dim(Vec<TDim, TIdx> const& vec) -> void
            {
                if constexpr(TDim::value > 0)
                {
                    for(auto i = std::min(typename TDim::value_type{3}, TDim::value); i < TDim::value; ++i)
                    {
                        if(vec[TDim::value - 1u - i] != 1)
                        {
                            throw std::runtime_error(
                                "The CUDA/HIP accelerator supports a maximum of 3 dimensions. All "
                                "work division extents of the dimensions higher 3 have to be 1!");
                        }
                    }
                }
            }

            template<typename TDim, typename TIdx>
            ALPAKA_FN_HOST auto convertVecToUniformCudaHipDim(Vec<TDim, TIdx> const& vec) -> dim3
            {
                dim3 dim(1, 1, 1);
                if constexpr(TDim::value >= 1)
                    dim.x = static_cast<unsigned>(vec[TDim::value - 1u]);
                if constexpr(TDim::value >= 2)
                    dim.y = static_cast<unsigned>(vec[TDim::value - 2u]);
                if constexpr(TDim::value >= 3)
                    dim.z = static_cast<unsigned>(vec[TDim::value - 3u]);
                checkVecOnly3Dim(vec);
                return dim;
            }
        } // namespace detail
    } // namespace uniform_cuda_hip

    //! The GPU CUDA/HIP accelerator execution task.
    template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelGpuUniformCudaHipRt final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelGpuUniformCudaHipRt(
            TWorkDiv&& workDiv,
            TKernelFnObj const& kernelFnObj,
            TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(kernelFnObj)
            , m_args(std::forward<TArgs>(args)...)
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        TKernelFnObj m_kernelFnObj;
        std::tuple<remove_restrict_t<std::decay_t<TArgs>>...> m_args;
    };

    namespace trait
    {
        //! The GPU CUDA/HIP execution task accelerator type trait specialization.
        template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccGpuUniformCudaHipRt<TApi, TDim, TIdx>;
        };

        //! The GPU CUDA/HIP execution task device type trait specialization.
        template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevUniformCudaHipRt<TApi>;
        };

        //! The GPU CUDA/HIP execution task dimension getter trait specialization.
        template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //! The CPU CUDA/HIP execution task platform type trait specialization.
        template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PlatformType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PlatformUniformCudaHipRt<TApi>;
        };

        //! The GPU CUDA/HIP execution task idx type trait specialization.
        template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };

        //! The CUDA/HIP non-blocking kernel enqueue trait specialization.
        template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking<TApi>,
            TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory idx

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // std::size_t printfFifoSize;
                // TApi::deviceGetLimit(&printfFifoSize, TApi::limitPrintfFifoSize);
                // std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                // TApi::deviceSetLimit(TApi::limitPrintfFifoSize, printfFifoSize*10);
                // TApi::deviceGetLimit(&printfFifoSize, TApi::limitPrintfFifoSize);
                // std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
#        endif
                auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(task);
                auto const blockThreadExtent = getWorkDiv<Block, Threads>(task);
                auto const threadElemExtent = getWorkDiv<Thread, Elems>(task);

                dim3 const gridDim = uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(gridBlockExtent);
                dim3 const blockDim = uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(blockThreadExtent);
                uniform_cuda_hip::detail::checkVecOnly3Dim(threadElemExtent);

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " gridDim: " << gridDim.z << " " << gridDim.y << " " << gridDim.x
                          << " blockDim: " << blockDim.z << " " << blockDim.y << " " << blockDim.x << std::endl;
#        endif

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                if(!isValidWorkDiv<TAcc>(getDev(queue), task))
                {
                    throw std::runtime_error(
                        "The given work division is not valid or not supported by the device of type "
                        + getAccName<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>() + "!");
                }
#        endif

                // Get the size of the block shared dynamic memory.
                auto const blockSharedMemDynSizeBytes = std::apply(
                    [&](remove_restrict_t<std::decay_t<TArgs>> const&... args) {
                        return getBlockSharedMemDynSizeBytes<TAcc>(
                            task.m_kernelFnObj,
                            blockThreadExtent,
                            threadElemExtent,
                            args...);
                    },
                    task.m_args);

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // Log the block shared memory idx.
                std::cout << __func__ << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                          << std::endl;
#        endif
                auto kernelName = alpaka::detail::
                    gpuKernel<TKernelFnObj, TApi, TAcc, TDim, TIdx, remove_restrict_t<std::decay_t<TArgs>>...>;

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // Log the function attributes.
                typename TApi::FuncAttributes_t funcAttrs;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::funcGetAttributes(&funcAttrs, kernelName));
                std::cout << __func__ << " binaryVersion: " << funcAttrs.binaryVersion
                          << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                          << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                          << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                          << " numRegs: " << funcAttrs.numRegs << " ptxVersion: " << funcAttrs.ptxVersion
                          << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B" << std::endl;
#        endif

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(queue.m_spQueueImpl->m_dev.getNativeHandle()));
                // Enqueue the kernel execution.
                // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch
                // language extension expects the arguments by value. This forces the type of a float argument given
                // with std::forward to this function to be of type float instead of e.g. "float const & __ptr64"
                // (MSVC). If not given by value, the kernel launch code does not copy the value but the pointer to the
                // value location.
                std::apply(
                    [&](remove_restrict_t<std::decay_t<TArgs>> const&... args)
                    {
                        kernelName<<<
                            gridDim,
                            blockDim,
                            static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                            queue.getNativeHandle()>>>(threadElemExtent, task.m_kernelFnObj, args...);
                    },
                    task.m_args);

                if constexpr(ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL)
                {
                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a
                    // custom error message.
                    std::ignore = TApi::streamSynchronize(queue.getNativeHandle());
                    auto const msg = std::string{
                        "'execution of kernel: '" + std::string{core::demangled<TKernelFnObj>} + "' failed with"};
                    ::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, true>(msg.c_str(), __FILE__, __LINE__);
                }
            }
        };

        //! The CUDA/HIP synchronous kernel enqueue trait specialization.
        template<typename TApi, typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking<TApi>,
            TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                TaskKernelGpuUniformCudaHipRt<TApi, TAcc, TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory idx

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // std::size_t printfFifoSize;
                // TApi::deviceGetLimit(&printfFifoSize, TApi::limitPrintfFifoSize);
                // std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                // TApi::deviceSetLimit(TApi::limitPrintfFifoSize, printfFifoSize*10);
                // TApi::deviceGetLimit(&printfFifoSize, TApi::limitPrintfFifoSize);
                // std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
#        endif
                auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(task);
                auto const blockThreadExtent = getWorkDiv<Block, Threads>(task);
                auto const threadElemExtent = getWorkDiv<Thread, Elems>(task);

                dim3 const gridDim = uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(gridBlockExtent);
                dim3 const blockDim = uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(blockThreadExtent);
                uniform_cuda_hip::detail::checkVecOnly3Dim(threadElemExtent);

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << "gridDim: " << gridDim.z << " " << gridDim.y << " " << gridDim.x << std::endl;
                std::cout << __func__ << "blockDim: " << blockDim.z << " " << blockDim.y << " " << blockDim.x
                          << std::endl;
#        endif

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                if(!isValidWorkDiv<TAcc>(getDev(queue), task))
                {
                    throw std::runtime_error(
                        "The given work division is not valid or not supported by the device of type "
                        + getAccName<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>>() + "!");
                }
#        endif

                // Get the size of the block shared dynamic memory.
                auto const blockSharedMemDynSizeBytes = std::apply(
                    [&](remove_restrict_t<std::decay_t<TArgs>> const&... args) {
                        return getBlockSharedMemDynSizeBytes<TAcc>(
                            task.m_kernelFnObj,
                            blockThreadExtent,
                            threadElemExtent,
                            args...);
                    },
                    task.m_args);

#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // Log the block shared memory idx.
                std::cout << __func__ << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                          << std::endl;
#        endif

                auto kernelName = alpaka::detail::
                    gpuKernel<TKernelFnObj, TApi, TAcc, TDim, TIdx, remove_restrict_t<std::decay_t<TArgs>>...>;
#        if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // Log the function attributes.
                typename TApi::FuncAttributes_t funcAttrs;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::funcGetAttributes(&funcAttrs, kernelName));
                std::cout << __func__ << " binaryVersion: " << funcAttrs.binaryVersion
                          << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                          << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                          << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                          << " numRegs: " << funcAttrs.numRegs << " ptxVersion: " << funcAttrs.ptxVersion
                          << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B" << std::endl;
#        endif

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(queue.m_spQueueImpl->m_dev.getNativeHandle()));

                // Enqueue the kernel execution.
                std::apply(
                    [&](remove_restrict_t<std::decay_t<TArgs>> const&... args)
                    {
                        kernelName<<<
                            gridDim,
                            blockDim,
                            static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                            queue.getNativeHandle()>>>(threadElemExtent, task.m_kernelFnObj, args...);
                    },
                    task.m_args);

                // Wait for the kernel execution to finish but do not check error return of this call.
                // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom
                // error message.
                std::ignore = TApi::streamSynchronize(queue.getNativeHandle());
                if constexpr(ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL)
                {
                    auto const msg
                        = std::string{"'execution of kernel: '" + core::demangled<TKernelFnObj> + "' failed with"};
                    ::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, true>(msg.c_str(), __FILE__, __LINE__);
                }
            }
        };
    } // namespace trait
} // namespace alpaka

#    endif

#endif
