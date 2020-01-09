/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
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

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccGpuHipRt.hpp>
#include <alpaka/dev/DevHipRt.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/queue/QueueHipRtBlocking.hpp>
#include <alpaka/queue/QueueHipRtNonBlocking.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <alpaka/acc/Traits.hpp>
    #include <alpaka/dev/Traits.hpp>
    #include <alpaka/workdiv/WorkDivHelpers.hpp>
#endif

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Hip.hpp>
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
        namespace hip
        {
            namespace detail
            {
                //-----------------------------------------------------------------------------
                //! The GPU HIP kernel entry point.
                // \NOTE: 'A __global__ function or function template cannot have a trailing return type.'
                template<
                    typename TDim,
                    typename TIdx,
                    typename TKernelFnObj,
                    typename... TArgs>
                __global__ void hipKernel(
                    hipLaunchParm lp,
                    vec::Vec<TDim, TIdx> const threadElemExtent,
                    TKernelFnObj const kernelFnObj,
                    TArgs ... args)
                {
#if BOOST_ARCH_PTX && (BOOST_ARCH_PTX < BOOST_VERSION_NUMBER(2, 0, 0))
    #error "Cuda device capability >= 2.0 is required!"
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
                    static_assert(
                        std::is_same<typename std::result_of<
                            TKernelFnObj(acc::AccGpuHipRt<TDim, TIdx> const &, TArgs const & ...)>::type, void>::value,
                        "The TKernelFnObj is required to return void!");
#pragma clang diagnostic pop

                    acc::AccGpuHipRt<TDim, TIdx> acc(threadElemExtent);

                    kernelFnObj(
                        const_cast<acc::AccGpuHipRt<TDim, TIdx> const &>(acc),
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
                ALPAKA_FN_HOST auto convertVecToHipDim(
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
        //! The GPU HIP accelerator execution task.
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelGpuHipRt final :
            public workdiv::WorkDivMembers<TDim, TIdx>
        {
        public:
// gcc-4.9 libstdc++ does not support std::is_trivially_copyable.
// MSVC std::is_trivially_copyable seems to be buggy (last tested at 15.7).
#if (!__GLIBCXX__) && (!BOOST_COMP_MSVC)
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
            //! Constructor.
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST TaskKernelGpuHipRt(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {
                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the execution task have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            TaskKernelGpuHipRt(TaskKernelGpuHipRt const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            TaskKernelGpuHipRt(TaskKernelGpuHipRt &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            auto operator=(TaskKernelGpuHipRt const &) -> TaskKernelGpuHipRt & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            auto operator=(TaskKernelGpuHipRt &&) -> TaskKernelGpuHipRt & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            ALPAKA_FN_HOST_ACC ~TaskKernelGpuHipRt() = default;

            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP execution task accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                kernel::TaskKernelGpuHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccGpuHipRt<TDim, TIdx>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP execution task device type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                kernel::TaskKernelGpuHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The GPU HIP execution task dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                kernel::TaskKernelGpuHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The CPU HIP execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct PltfType<
                kernel::TaskKernelGpuHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The GPU HIP execution task idx type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct IdxType<
                kernel::TaskKernelGpuHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
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
            //! The HIP non-blocking kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueHipRtNonBlocking,
                kernel::TaskKernelGpuHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------

                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueHipRtNonBlocking & queue,
                    kernel::TaskKernelGpuHipRt<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory size

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //hipDeviceGetLimit(&printfFifoSize, hipLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //hipDeviceSetLimit(hipLimitPrintfFifoSize, printfFifoSize*10);
                    //hipDeviceGetLimit(&printfFifoSize, hipLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

                    dim3 const gridDim(kernel::hip::detail::convertVecToHipDim(gridBlockExtent));
                    dim3 const blockDim(kernel::hip::detail::convertVecToHipDim(blockThreadExtent));
                    kernel::hip::detail::checkVecOnly3Dim(threadElemExtent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__
                        << " gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x
                        << " blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x
                        << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuHipRt<TDim, TIdx>>(dev::getDev(queue), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuHipRt<TDim, TIdx>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            // workaround for HIP(HCC) to
                            // avoid forbidden host-call
                            // within host-device functions
                            #if defined(BOOST_COMP_HCC) && BOOST_COMP_HCC
                            ALPAKA_FN_HOST_ACC
                            #endif
                            [&](TArgs const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuHipRt<TDim, TIdx>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory size.
                    std::cout << __func__
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the function attributes.
                    /*hipFuncAttributes funcAttrs;
                    hipFuncGetAttributes(&funcAttrs, kernel::hip::detail::hipKernel<TDim, TIdx, TKernelFnObj, TArgs...>);
                    std::cout << __func__
                        << " binaryVersion: " << funcAttrs.binaryVersion
                        << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                        << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                        << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                        << " numRegs: " << funcAttrs.numRegs
                        << " ptxVersion: " << funcAttrs.ptxVersion
                        << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B"
                        << std::endl; */
#endif

                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    meta::apply(
                        [&](TArgs ... args)
                        {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(kernel::hip::detail::hipKernel< TDim, TIdx, TKernelFnObj, TArgs... >),
                                gridDim,
                                blockDim,
                                static_cast<std::uint32_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_HipQueue,
                                hipLaunchParm{},
                                threadElemExtent,
                                task.m_kernelFnObj,
                                std::move(args)...
                            );

                        },
                        task.m_args);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    hipStreamSynchronize(
                        queue.m_spQueueImpl->m_HipQueue);
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                    ::alpaka::hip::detail::hipRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
#endif
                }
            };
            //#############################################################################
            //! The HIP synchronous kernel enqueue trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueHipRtBlocking,
                kernel::TaskKernelGpuHipRt<TDim, TIdx, TKernelFnObj, TArgs...>>
            {
                //-----------------------------------------------------------------------------

                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueHipRtBlocking & queue,
                    kernel::TaskKernelGpuHipRt<TDim, TIdx, TKernelFnObj, TArgs...> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // TODO: Check that (sizeof(TKernelFnObj) * m_3uiBlockThreadExtent.prod()) < available memory size

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t printfFifoSize;
                    //hipDeviceGetLimit(&printfFifoSize, hipLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " << printfFifoSize << std::endl;
                    //hipDeviceSetLimit(hipLimitPrintfFifoSize, printfFifoSize*10);
                    //hipDeviceGetLimit(&printfFifoSize, hipLimitPrintfFifoSize);
                    //std::cout << __func__ << "INFO: printfFifoSize: " <<  printfFifoSize << std::endl;
#endif
                    auto const gridBlockExtent(
                        workdiv::getWorkDiv<Grid, Blocks>(task));
                    auto const blockThreadExtent(
                        workdiv::getWorkDiv<Block, Threads>(task));
                    auto const threadElemExtent(
                        workdiv::getWorkDiv<Thread, Elems>(task));

                    dim3 gridDim(kernel::hip::detail::convertVecToHipDim(gridBlockExtent));
                    dim3 blockDim(kernel::hip::detail::convertVecToHipDim(blockThreadExtent));
                    kernel::hip::detail::checkVecOnly3Dim(threadElemExtent);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__ << "gridDim: " <<  gridDim.z << " " <<  gridDim.y << " " <<  gridDim.x << std::endl;
                    std::cout << __func__ << "blockDim: " <<  blockDim.z << " " <<  blockDim.y << " " <<  blockDim.x << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // This checks for a valid work division that is also compliant with the maxima of the accelerator.
                    if(!workdiv::isValidWorkDiv<acc::AccGpuHipRt<TDim, TIdx>>(dev::getDev(queue), task))
                    {
                        throw std::runtime_error("The given work division is not valid or not supported by the device of type " + acc::getAccName<acc::AccGpuHipRt<TDim, TIdx>>() + "!");
                    }
#endif

                    // Get the size of the block shared dynamic memory.
                    auto const blockSharedMemDynSizeBytes(
                        meta::apply(
                            [&](TArgs const & ... args)
                            {
                                return
                                    kernel::getBlockSharedMemDynSizeBytes<
                                        acc::AccGpuHipRt<TDim, TIdx>>(
                                            task.m_kernelFnObj,
                                            blockThreadExtent,
                                            threadElemExtent,
                                            args...);
                            },
                            task.m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // Log the block shared memory size.
                    std::cout << __func__
                        << " BlockSharedMemDynSizeBytes: " << blockSharedMemDynSizeBytes << " B" << std::endl;
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    // hipFuncAttributes not ported from HIP to HIP.
                    // Log the function attributes.
                    /*hipFuncAttributes funcAttrs;
                    hipFuncGetAttributes(&funcAttrs, kernel::hip::detail::hipKernel<TDim, TIdx, TKernelFnObj, TArgs...>);
                    std::cout << __func__
                        << " binaryVersion: " << funcAttrs.binaryVersion
                        << " constSizeBytes: " << funcAttrs.constSizeBytes << " B"
                        << " localSizeBytes: " << funcAttrs.localSizeBytes << " B"
                        << " maxThreadsPerBlock: " << funcAttrs.maxThreadsPerBlock
                        << " numRegs: " << funcAttrs.numRegs
                        << " ptxVersion: " << funcAttrs.ptxVersion
                        << " sharedSizeBytes: " << funcAttrs.sharedSizeBytes << " B"
                        << std::endl;*/
#endif

                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            queue.m_spQueueImpl->m_dev.m_iDevice));
                    // Enqueue the kernel execution.
                    // \NOTE: No const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    meta::apply(
                        [&](TArgs ... args)
                        {
                            hipLaunchKernel(
                                HIP_KERNEL_NAME(kernel::hip::detail::hipKernel< TDim, TIdx, TKernelFnObj, TArgs... >),
                                gridDim,
                                blockDim,
                                static_cast<std::size_t>(blockSharedMemDynSizeBytes),
                                queue.m_spQueueImpl->m_HipQueue,
                                threadElemExtent,
                                task.m_kernelFnObj,
                                args...
                            );
                        },
                        task.m_args);

                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    hipStreamSynchronize(
                        queue.m_spQueueImpl->m_HipQueue);
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    std::string const kernelName("'execution of kernel: '" + std::string(typeid(TKernelFnObj).name()) + "' failed with");
                    ::alpaka::hip::detail::hipRtCheckLastError(kernelName.c_str(), __FILE__, __LINE__);
#endif
                }
            };
        }
    }
}

#endif
