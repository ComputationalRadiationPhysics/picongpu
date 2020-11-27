/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/queue/Traits.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

// Implementation details.
#    include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>
#    include <alpaka/workdiv/WorkDivMembers.hpp>

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <alpaka/acc/Traits.hpp>
#        include <alpaka/dev/Traits.hpp>
#        include <alpaka/workdiv/WorkDivHelpers.hpp>
#    endif

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/meta/ApplyTuple.hpp>
#    include <alpaka/meta/Metafunctions.hpp>

#    include <stdexcept>
#    include <tuple>
#    include <type_traits>
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#        include <iostream>
#    endif

namespace alpaka
{
    namespace uniform_cuda_hip
    {
        namespace detail
        {
            //-----------------------------------------------------------------------------
            //! The GPU CUDA/HIP kernel entry point.
            // \NOTE: 'A __global__ function or function template cannot have a trailing return type.'
            template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
            __global__ void uniformCudaHipKernel(
                Vec<TDim, TIdx> const threadElemExtent,
                TKernelFnObj const kernelFnObj,
                TArgs... args)
            {
#    if BOOST_ARCH_PTX && (BOOST_ARCH_PTX < BOOST_VERSION_NUMBER(2, 0, 0))
#        error "Device capability >= 2.0 is required!"
#    endif

                const TAcc acc(threadElemExtent);

// with clang it is not possible to query std::result_of for a pure device lambda created on the host side
#    if !(BOOST_COMP_CLANG_CUDA && BOOST_COMP_CLANG)
                static_assert(
                    std::is_same<decltype(kernelFnObj(const_cast<TAcc const&>(acc), args...)), void>::value,
                    "The TKernelFnObj is required to return void!");
#    endif
                kernelFnObj(const_cast<TAcc const&>(acc), args...);
            }

            //-----------------------------------------------------------------------------
            template<typename TDim, typename TIdx>
            ALPAKA_FN_HOST auto checkVecOnly3Dim(Vec<TDim, TIdx> const& vec) -> void
            {
                for(auto i(std::min(static_cast<typename TDim::value_type>(3), TDim::value)); i < TDim::value; ++i)
                {
                    if(vec[TDim::value - 1u - i] != 1)
                    {
                        throw std::runtime_error("The CUDA/HIP accelerator supports a maximum of 3 dimensions. All "
                                                 "work division extents of the dimensions higher 3 have to be 1!");
                    }
                }
            }

            //-----------------------------------------------------------------------------
            template<typename TDim, typename TIdx>
            ALPAKA_FN_HOST auto convertVecToUniformCudaHipDim(Vec<TDim, TIdx> const& vec) -> dim3
            {
                dim3 dim(1, 1, 1);
                for(auto i(static_cast<typename TDim::value_type>(0));
                    i < std::min(static_cast<typename TDim::value_type>(3), TDim::value);
                    ++i)
                {
                    reinterpret_cast<unsigned int*>(&dim)[i] = static_cast<unsigned int>(vec[TDim::value - 1u - i]);
                }
                checkVecOnly3Dim(vec);
                return dim;
            }
        } // namespace detail
    } // namespace uniform_cuda_hip

    //#############################################################################
    //! The GPU CUDA/HIP accelerator execution task.
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelGpuUniformCudaHipRt final : public WorkDivMembers<TDim, TIdx>
    {
    public:
        //-----------------------------------------------------------------------------
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
        //-----------------------------------------------------------------------------
        TaskKernelGpuUniformCudaHipRt(TaskKernelGpuUniformCudaHipRt const&) = default;
        //-----------------------------------------------------------------------------
        TaskKernelGpuUniformCudaHipRt(TaskKernelGpuUniformCudaHipRt&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelGpuUniformCudaHipRt const&) -> TaskKernelGpuUniformCudaHipRt& = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelGpuUniformCudaHipRt&&) -> TaskKernelGpuUniformCudaHipRt& = default;
        //-----------------------------------------------------------------------------
        ~TaskKernelGpuUniformCudaHipRt() = default;

        TKernelFnObj m_kernelFnObj;
        std::tuple<std::decay_t<TArgs>...> m_args;
    };

    namespace traits
    {
        //#############################################################################
        //! The GPU CUDA/HIP execution task accelerator type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelGpuUniformCudaHipRt<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccGpuUniformCudaHipRt<TDim, TIdx>;
        };

        //#############################################################################
        //! The GPU CUDA/HIP execution task device type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelGpuUniformCudaHipRt<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevUniformCudaHipRt;
        };

        //#############################################################################
        //! The GPU CUDA/HIP execution task dimension getter trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelGpuUniformCudaHipRt<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The CPU CUDA/HIP execution task platform type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelGpuUniformCudaHipRt<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfUniformCudaHipRt;
        };

        //#############################################################################
        //! The GPU CUDA/HIP execution task idx type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelGpuUniformCudaHipRt<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };

        //#############################################################################
        //! The CUDA/HIP non-blocking kernel enqueue trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking,
            TaskKernelGpuUniformCudaHipRt<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking& queue,
                TaskKernelGpuUniformCudaHipRt<TAcc, TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory idx

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // std::size_t printfFifoSize;
                // cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                // std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                // cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                // std::cout << __func__ << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#    endif
                auto const gridBlockExtent(getWorkDiv<Grid, Blocks>(task));
                auto const blockThreadExtent(getWorkDiv<Block, Threads>(task));
                auto const threadElemExtent(getWorkDiv<Thread, Elems>(task));

                dim3 const gridDim(uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(gridBlockExtent));
                dim3 const blockDim(uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(blockThreadExtent));
                uniform_cuda_hip::detail::checkVecOnly3Dim(threadElemExtent);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " gridDim: " << gridDim.z << " " << gridDim.y << " " << gridDim.x
                          << " blockDim: " << blockDim.z << " " << blockDim.y << " " << blockDim.x << std::endl;
#    endif

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                if(!isValidWorkDiv<TAcc>(getDev(queue), task))
                {
                    throw std::runtime_error(
                        "The given work division is not valid or not supported by the device of type "
                        + getAccName<AccGpuUniformCudaHipRt<TDim, TIdx>>() + "!");
                }
#    endif

                // Get the size of the block shared dynamic memory.
                auto const blockSharedMemDynSizeBytes(meta::apply(
                    [&](ALPAKA_DECAY_T(TArgs) const&... args) {
                        return getBlockSharedMemDynSizeBytes<TAcc>(
                            task.m_kernelFnObj,
                            blockThreadExtent,
                            threadElemExtent,
                            args...);
                    },
                    task.m_args));

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // Log the block shared memory idx.
                std::cout << __func__ << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                          << std::endl;
#    endif
                auto kernelName = uniform_cuda_hip::detail::
                    uniformCudaHipKernel<TAcc, TDim, TIdx, TKernelFnObj, std::decay_t<TArgs>...>;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

                // Log the function attributes.
                cudaFuncAttributes funcAttrs;
                cudaFuncGetAttributes(&funcAttrs, kernelName);
                std::cout << __func__ << " binaryVersion: " << funcAttrs.binaryVersion
                          << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                          << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                          << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                          << " numRegs: " << funcAttrs.numRegs << " ptxVersion: " << funcAttrs.ptxVersion
                          << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B" << std::endl;
#        endif
#    endif

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(queue.m_spQueueImpl->m_dev.m_iDevice));
                // Enqueue the kernel execution.
                // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch
                // language extension expects the arguments by value. This forces the type of a float argument given
                // with std::forward to this function to be of type float instead of e.g. "float const & __ptr64"
                // (MSVC). If not given by value, the kernel launch code does not copy the value but the pointer to the
                // value location.
                meta::apply(
                    [&](ALPAKA_DECAY_T(TArgs) const&... args) {
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        kernelName<<<
                            gridDim,
                            blockDim,
                            static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                            queue.m_spQueueImpl->m_UniformCudaHipQueue>>>(
                            threadElemExtent,
                            task.m_kernelFnObj,
                            args...);
#    else
                        hipLaunchKernelGGL(
                            HIP_KERNEL_NAME(kernelName),
                            gridDim,
                            blockDim,
                            static_cast<std::uint32_t>(blockSharedMemDynSizeBytes),
                            queue.m_spQueueImpl->m_UniformCudaHipQueue,
                            threadElemExtent,
                            task.m_kernelFnObj,
                            args...);
#    endif
                    },
                    task.m_args);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                // Wait for the kernel execution to finish but do not check error return of this call.
                // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom
                // error message.
                ALPAKA_API_PREFIX(StreamSynchronize)(queue.m_spQueueImpl->m_UniformCudaHipQueue);
                std::string const msg(
                    "'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                ::alpaka::uniform_cuda_hip::detail::rtCheckLastError(msg.c_str(), __FILE__, __LINE__);
#    endif
            }
        };
        //#############################################################################
        //! The CUDA/HIP synchronous kernel enqueue trait specialization.
        template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking,
            TaskKernelGpuUniformCudaHipRt<TAcc, TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking& queue,
                TaskKernelGpuUniformCudaHipRt<TAcc, TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory idx

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // std::size_t printfFifoSize;
                // cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                // std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                // cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                // std::cout << __func__ << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#    endif
                auto const gridBlockExtent(getWorkDiv<Grid, Blocks>(task));
                auto const blockThreadExtent(getWorkDiv<Block, Threads>(task));
                auto const threadElemExtent(getWorkDiv<Thread, Elems>(task));

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                dim3 const gridDim(uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(gridBlockExtent));
                dim3 const blockDim(uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(blockThreadExtent));
                uniform_cuda_hip::detail::checkVecOnly3Dim(threadElemExtent);
#    else
                dim3 gridDim(uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(gridBlockExtent));
                dim3 blockDim(uniform_cuda_hip::detail::convertVecToUniformCudaHipDim(blockThreadExtent));
                uniform_cuda_hip::detail::checkVecOnly3Dim(threadElemExtent);
#    endif


#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << "gridDim: " << gridDim.z << " " << gridDim.y << " " << gridDim.x << std::endl;
                std::cout << __func__ << "blockDim: " << blockDim.z << " " << blockDim.y << " " << blockDim.x
                          << std::endl;
#    endif

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                if(!isValidWorkDiv<TAcc>(getDev(queue), task))
                {
                    throw std::runtime_error(
                        "The given work division is not valid or not supported by the device of type "
                        + getAccName<AccGpuUniformCudaHipRt<TDim, TIdx>>() + "!");
                }
#    endif

                // Get the size of the block shared dynamic memory.
                auto const blockSharedMemDynSizeBytes(meta::apply(
                    [&](ALPAKA_DECAY_T(TArgs) const&... args) {
                        return getBlockSharedMemDynSizeBytes<TAcc>(
                            task.m_kernelFnObj,
                            blockThreadExtent,
                            threadElemExtent,
                            args...);
                    },
                    task.m_args));

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // Log the block shared memory idx.
                std::cout << __func__ << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B"
                          << std::endl;
#    endif

                auto kernelName = uniform_cuda_hip::detail::
                    uniformCudaHipKernel<TAcc, TDim, TIdx, TKernelFnObj, std::decay_t<TArgs>...>;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                // hipFuncAttributes not ported from HIP to HIP.
                // TODO why this is currently not possible
                //
                // Log the function attributes.
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                ALPAKA_API_PREFIX(FuncAttributes) funcAttrs;
                ALPAKA_API_PREFIX(FuncGetAttributes)(&funcAttrs, kernelName);
                std::cout << __func__ << " binaryVersion: " << funcAttrs.binaryVersion
                          << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                          << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                          << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                          << " numRegs: " << funcAttrs.numRegs << " ptxVersion: " << funcAttrs.ptxVersion
                          << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B" << std::endl;
#        endif
#    endif

                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(queue.m_spQueueImpl->m_dev.m_iDevice));

                // Enqueue the kernel execution.
                meta::apply(
                    [&](ALPAKA_DECAY_T(TArgs) const&... args) {
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        kernelName<<<
                            gridDim,
                            blockDim,
                            static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                            queue.m_spQueueImpl->m_UniformCudaHipQueue>>>(
                            threadElemExtent,
                            task.m_kernelFnObj,
                            args...);
#    else
                        hipLaunchKernelGGL(
                            HIP_KERNEL_NAME(kernelName),
                            gridDim,
                            blockDim,
                            static_cast<std::uint32_t>(blockSharedMemDynSizeBytes),
                            queue.m_spQueueImpl->m_UniformCudaHipQueue,
                            threadElemExtent,
                            task.m_kernelFnObj,
                            args...);
#    endif
                    },
                    task.m_args);

                // Wait for the kernel execution to finish but do not check error return of this call.
                // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom
                // error message.
                ALPAKA_API_PREFIX(StreamSynchronize)(queue.m_spQueueImpl->m_UniformCudaHipQueue);
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                std::string const msg(
                    "'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                ::alpaka::uniform_cuda_hip::detail::rtCheckLastError(msg.c_str(), __FILE__, __LINE__);
#    endif
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
