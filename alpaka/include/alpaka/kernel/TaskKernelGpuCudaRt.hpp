/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
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

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccGpuCudaRt.hpp>
#include <alpaka/dev/DevCudaRt.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/queue/QueueCudaRtNonBlocking.hpp>
#include <alpaka/queue/QueueCudaRtBlocking.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <alpaka/acc/Traits.hpp>
    #include <alpaka/dev/Traits.hpp>
    #include <alpaka/workdiv/WorkDivHelpers.hpp>
#endif

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Cuda.hpp>
#include <alpaka/meta/ApplyTuple.hpp>
#include <alpaka/meta/Metafunctions.hpp>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace kernel
    {
        namespace cuda
        {
            namespace detail
            {
                //-----------------------------------------------------------------------------
                //! The GPU CUDA kernel entry point.
                // \NOTE: 'A __global__ function or function template cannot have a trailing return type.'
                template<
                    typename TDim,
                    typename TIdx,
                    typename TKernelFnObj,
                    typename... TArgs>
                __global__ void cudaKernel(
                    vec::Vec<TDim, TIdx> const threadElemExtent,
                    TKernelFnObj const kernelFnObj,
                    TArgs ... args)
                {
#if BOOST_ARCH_PTX && (BOOST_ARCH_PTX < BOOST_VERSION_NUMBER(2, 0, 0))
    #error "Cuda device capability >= 2.0 is required!"
#endif

// with clang it is not possible to query std::result_of for a pure device lambda created on the host side
#if !(BOOST_COMP_CLANG_CUDA && BOOST_COMP_CLANG)
                    static_assert(
                        std::is_same<typename std::result_of<
                            TKernelFnObj(acc::AccGpuCudaRt<TDim, TIdx> const &, TArgs const & ...)>::type, void>::value,
                        "The TKernelFnObj is required to return void!");
#endif
                    acc::AccGpuCudaRt<TDim, TIdx> acc(threadElemExtent);

                    kernelFnObj(
                        const_cast<acc::AccGpuCudaRt<TDim, TIdx> const &>(acc),
                        args...);
                }

                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename TIdx
                >
                ALPAKA_FN_HOST auto checkVecOnly3Dim(
                    vec::Vec<TDim, TIdx> const & vec)
                -> void
                {
                    for(auto i(std::min(static_cast<typename TDim::value_type>(3), TDim::value)); i<TDim::value; ++i)
                    {
                        if(vec[TDim::value-1u-i] != 1)
                        {
                            throw std::runtime_error("The CUDA accelerator supports a maximum of 3 dimensions. All work division extents of the dimensions higher 3 have to be 1!");
                        }
                    }
                }

                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename TIdx
                >
                ALPAKA_FN_HOST auto convertVecToCudaDim(
                    vec::Vec<TDim, TIdx> const & vec)
                -> dim3
                {
                    dim3 dim(1, 1, 1);
                    for(auto i(static_cast<typename TDim::value_type>(0)); i<std::min(static_cast<typename TDim::value_type>(3), TDim::value); ++i)
                    {
                        reinterpret_cast<unsigned int *>(&dim)[i] = static_cast<unsigned int>(vec[TDim::value-1u-i]);
                    }
                    checkVecOnly3Dim(vec);
                    return dim;
                }
            }
        }

        //#############################################################################
        //! The GPU CUDA accelerator execution task.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelGpuCudaRt final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
// gcc-4.9 libstdc++ does not support std::is_trivially_copyable.
// MSVC std::is_trivially_copyable seems to be buggy (last tested at 15.7).
// libc++ in combination with CUDA does not seem to work.
#if (!BOOST_COMP_MSVC) && !(defined(__GLIBCXX__) && (__GLIBCXX__)) && !(defined(_LIBCPP_VERSION) && BOOST_LANG_CUDA)
            static_assert(
                meta::Conjunction<
                    std::is_trivially_copyable<
                        TKernelFnObj>,
                    std::is_trivially_copyable<
                        TArgs>...
                    >::value,
                "The given kernel function object and its arguments have to fulfill is_trivially_copyable!");
#endif

            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelGpuCudaRt(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs && ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(std::forward<TArgs>(args)...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            TaskKernelGpuCudaRt(TaskKernelGpuCudaRt const &) = default;
            //-----------------------------------------------------------------------------
            TaskKernelGpuCudaRt(TaskKernelGpuCudaRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelGpuCudaRt const &) -> TaskKernelGpuCudaRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(TaskKernelGpuCudaRt &&) -> TaskKernelGpuCudaRt & = default;
            //-----------------------------------------------------------------------------
            ~TaskKernelGpuCudaRt() = default;

            TKernelFnObj m_kernelFnObj;
            std::tuple<typename std::decay<TArgs>::type...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA execution task accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccGpuCudaRt<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA execution task device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The GPU CUDA execution task dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
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
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The GPU CUDA execution task idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = TIdx;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA non-blocking kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueCudaRtNonBlocking,
                kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtNonBlocking & queue,
                    kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory idx

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

                    dim3 const gridDim(kernel::cuda::detail::convertVecToCudaDim(gridBlockExtent));
                    dim3 const blockDim(kernel::cuda::detail::convertVecToCudaDim(blockThreadExtent));
                    kernel::cuda::detail::checkVecOnly3Dim(threadElemExtent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__
                        << " gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x
                        << " blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x
                        << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuCudaRt<TDim, TIdx>>(dev::getDev(queue), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaRt<TDim, TIdx>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            [&](typename std::decay<TArgs>::type const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuCudaRt<TDim, TIdx>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory idx.
                    std::cout << __func__
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, kernel::cuda::detail::cudaKernel<TDim, TIdx, TKernelFnObj, TArgs...>);
                    std::cout << __func__
                        << " binaryVersion: " << funcAttrs.binaryVersion
                        << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                        << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                        << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                        << " numRegs: " << funcAttrs.numRegs
                        << " ptxVersion: " << funcAttrs.ptxVersion
                        << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B"
                        << std::endl;
#endif

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    meta::apply(
                        [&](typename std::decay<TArgs>::type const & ... args)
                        {
                            kernel::cuda::detail::cudaKernel<TDim, TIdx, TKernelFnObj, typename std::decay<TArgs>::type...><<<
                                gridDim,
                                blockDim,
                                static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_CudaQueue>>>(
                                    threadElemExtent,
                                    task.m_kernelFnObj,
                                    args...);
                        },
                        task.m_args);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(
                        queue.m_spQueueImpl->m_CudaQueue);
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                    ::alpaka::cuda::detail::cudaRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
#endif
                }
            };
            //#############################################################################
            //! The CUDA synchronous kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueCudaRtBlocking,
                kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtBlocking & queue,
                    kernel::TaskKernelGpuCudaRt<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory idx

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

                    dim3 const gridDim(kernel::cuda::detail::convertVecToCudaDim(gridBlockExtent));
                    dim3 const blockDim(kernel::cuda::detail::convertVecToCudaDim(blockThreadExtent));
                    kernel::cuda::detail::checkVecOnly3Dim(threadElemExtent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__ << "gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x << std::endl;
                    std::cout << __func__ << "blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuCudaRt<TDim, TIdx>>(dev::getDev(queue), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaRt<TDim, TIdx>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            [&](typename std::decay<TArgs>::type const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuCudaRt<TDim, TIdx>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory idx.
                    std::cout << __func__
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, kernel::cuda::detail::cudaKernel<TDim, TIdx, TKernelFnObj, typename std::decay<TArgs>::type...>);
                    std::cout << __func__
                        << " binaryVersion: " << funcAttrs.binaryVersion
                        << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                        << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                        << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                        << " numRegs: " << funcAttrs.numRegs
                        << " ptxVersion: " << funcAttrs.ptxVersion
                        << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B"
                        << std::endl;
#endif

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    meta::apply(
                        [&](typename std::decay<TArgs>::type const & ... args)
                        {
                            kernel::cuda::detail::cudaKernel<TDim, TIdx, TKernelFnObj, typename std::decay<TArgs>::type...><<<
                                gridDim,
                                blockDim,
                                static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_CudaQueue>>>(
                                    threadElemExtent,
                                    task.m_kernelFnObj,
                                    args...);
                        },
                        task.m_args);

                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(
                        queue.m_spQueueImpl->m_CudaQueue);
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                    ::alpaka::cuda::detail::cudaRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
#endif
                }
            };
        }
    }
}

#endif
