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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/exec/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/size/Traits.hpp>
#include <alpaka/stream/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccGpuCudaRt.hpp>
#include <alpaka/dev/DevCudaRt.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/stream/StreamCudaRtAsync.hpp>
#include <alpaka/stream/StreamCudaRtSync.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <alpaka/acc/Traits.hpp>
    #include <alpaka/dev/Traits.hpp>
    #include <alpaka/workdiv/WorkDivHelpers.hpp>
#endif

#include <alpaka/core/Cuda.hpp>
#include <alpaka/meta/ApplyTuple.hpp>
#include <alpaka/meta/Metafunctions.hpp>

#include <boost/predef.h>
#include <boost/assert.hpp>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

namespace alpaka
{
    namespace exec
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
                    typename TSize,
                    typename TKernelFnObj,
                    typename... TArgs>
                __global__ void cudaKernel(
                    vec::Vec<TDim, TSize> const threadElemExtent,
                    TKernelFnObj const kernelFnObj,
                    TArgs ... args)
                {
#if BOOST_ARCH_CUDA_DEVICE && (BOOST_ARCH_CUDA_DEVICE < BOOST_VERSION_NUMBER(2, 0, 0))
    #error "Cuda device capability >= 2.0 is required!"
#endif
                    acc::AccGpuCudaRt<TDim, TSize> acc(threadElemExtent);

                    kernelFnObj(
                        const_cast<acc::AccGpuCudaRt<TDim, TSize> const &>(acc),
                        args...);
                }
            }
        }

        //#############################################################################
        //! The GPU CUDA accelerator executor.
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecGpuCudaRt final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
#if (!__GLIBCXX__) // libstdc++ even for gcc-4.9 does not support std::is_trivially_copyable.
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
            ALPAKA_FN_HOST ExecGpuCudaRt(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TSize>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            ExecGpuCudaRt(ExecGpuCudaRt const &) = default;
            //-----------------------------------------------------------------------------
            ExecGpuCudaRt(ExecGpuCudaRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecGpuCudaRt const &) -> ExecGpuCudaRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(ExecGpuCudaRt &&) -> ExecGpuCudaRt & = default;
            //-----------------------------------------------------------------------------
            ~ExecGpuCudaRt() = default;

            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA executor accelerator type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccGpuCudaRt<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA executor device type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The GPU CUDA executor dimension getter trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The GPU CUDA executor executor type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU CUDA executor platform type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = pltf::PltfCudaRt;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA executor size type trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TSize;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA asynchronous kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory size

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << BOOST_CURRENT_FUNCTION << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << BOOST_CURRENT_FUNCTION << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

                    dim3 gridDim(1u, 1u, 1u);
                    dim3 blockDim(1u, 1u, 1u);
                    // \FIXME: CUDA currently supports a maximum of 3 dimensions!
                    for(auto i(static_cast<typename TDim::value_type>(0)); i<std::min(static_cast<typename TDim::value_type>(3), TDim::value); ++i)
                    {
                        reinterpret_cast<unsigned int *>(&gridDim)[i] = static_cast<unsigned int>(gridBlockExtent[TDim::value-1u-i]);
                        reinterpret_cast<unsigned int *>(&blockDim)[i] = static_cast<unsigned int>(blockThreadExtent[TDim::value-1u-i]);

                    }
                    // Assert that all extent of the higher dimensions are 1!
                    for(auto i(std::min(static_cast<typename TDim::value_type>(3), TDim::value)); i<TDim::value; ++i)
                    {
                        BOOST_VERIFY(gridBlockExtent[TDim::value-1u-i] == 1);
                        BOOST_VERIFY(blockThreadExtent[TDim::value-1u-i] == 1);
                        BOOST_VERIFY(threadElemExtent[TDim::value-1u-i] == 1);
                    }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x
                        << " blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x
                        << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuCudaRt<TDim, TSize>>(dev::getDev(stream), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaRt<TDim, TSize>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            [&](TArgs const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuCudaRt<TDim, TSize>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory size.
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, exec::cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...>);
                    std::cout << BOOST_CURRENT_FUNCTION
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
                            stream.m_spStreamImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    meta::apply(
                        [&](TArgs ... args)
                        {
                            exec::cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...><<<
                                gridDim,
                                blockDim,
                                static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                                stream.m_spStreamImpl->m_CudaStream>>>(
                                    threadElemExtent,
                                    task.m_kernelFnObj,
                                    args...);
                        },
                        task.m_args);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(
                        stream.m_spStreamImpl->m_CudaStream);
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                    ::alpaka::cuda::detail::cudaRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
#endif
                }
            };
            //#############################################################################
            //! The CUDA synchronous kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                stream::StreamCudaRtSync,
                exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync & stream,
                    exec::ExecGpuCudaRt<TDim, TSize, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory size

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << BOOST_CURRENT_FUNCTION << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfFifoSize*10);
                    //cudaDeviceGetLimit(&printfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << BOOST_CURRENT_FUNCTION << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

                    dim3 gridDim(1u, 1u, 1u);
                    dim3 blockDim(1u, 1u, 1u);
                    // \FIXME: CUDA currently supports a maximum of 3 dimensions!
                    for(auto i(static_cast<typename TDim::value_type>(0)); i<std::min(static_cast<typename TDim::value_type>(3), TDim::value); ++i)
                    {
                        reinterpret_cast<unsigned int *>(&gridDim)[i] = static_cast<unsigned int>(gridBlockExtent[TDim::value-1u-i]);
                        reinterpret_cast<unsigned int *>(&blockDim)[i] = static_cast<unsigned int>(blockThreadExtent[TDim::value-1u-i]);
                    }
                    // Assert that all extent of the higher dimensions are 1!
                    for(auto i(std::min(static_cast<typename TDim::value_type>(3), TDim::value)); i<TDim::value; ++i)
                    {
                        BOOST_VERIFY(gridBlockExtent[TDim::value-1u-i] == 1);
                        BOOST_VERIFY(blockThreadExtent[TDim::value-1u-i] == 1);
                        BOOST_VERIFY(threadElemExtent[TDim::value-1u-i] == 1);
                    }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION << "gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x << std::endl;
                    std::cout << BOOST_CURRENT_FUNCTION << "blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuCudaRt<TDim, TSize>>(dev::getDev(stream), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuCudaRt<TDim, TSize>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            [&](TArgs const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuCudaRt<TDim, TSize>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory size.
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
                    cudaFuncAttributes funcAttrs;
                    cudaFuncGetAttributes(&funcAttrs, exec::cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...>);
                    std::cout << BOOST_CURRENT_FUNCTION
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
                            stream.m_spStreamImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    meta::apply(
                        [&](TArgs ... args)
                        {
                            exec::cuda::detail::cudaKernel<TDim, TSize, TKernelFnObj, TArgs...><<<
                                gridDim,
                                blockDim,
                                blockSharedMemDynSizeBytes,
                                stream.m_spStreamImpl->m_CudaStream>>>(
                                    threadElemExtent,
                                    task.m_kernelFnObj,
                                    args...);
                        },
                        task.m_args);

                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(
                        stream.m_spStreamImpl->m_CudaStream);
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
