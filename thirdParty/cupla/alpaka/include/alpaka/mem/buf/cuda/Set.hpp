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

#include <alpaka/queue/QueueCudaRtBlocking.hpp>
#include <alpaka/queue/QueueCudaRtNonBlocking.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Cuda.hpp>


namespace alpaka
{
    namespace dev
    {
        class DevCudaRt;
    }
}

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace cuda
            {
                namespace detail
                {
                    //#############################################################################
                    //! The CUDA memory set trait.
                    template<
                        typename TDim,
                        typename TView,
                        typename TExtent>
                    struct TaskSetCuda
                    {
                        //-----------------------------------------------------------------------------
                        TaskSetCuda(
                            TView & view,
                            std::uint8_t const & byte,
                            TExtent const & extent) :
                                m_view(view),
                                m_byte(byte),
                                m_extent(extent),
                                m_iDevice(dev::getDev(view).m_iDevice)
                        {
                            static_assert(
                                !std::is_const<TView>::value,
                                "The destination view can not be const!");

                            static_assert(
                                dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                                "The destination view and the extent are required to have the same dimensionality!");
                        }

                        TView & m_view;
                        std::uint8_t const m_byte;
                        TExtent const m_extent;
                        std::int32_t const m_iDevice;
                    };
                }
            }
            namespace traits
            {
                //#############################################################################
                //! The CUDA device memory set trait specialization.
                template<
                    typename TDim>
                struct CreateTaskSet<
                    TDim,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TView>
                    ALPAKA_FN_HOST static auto createTaskSet(
                        TView & view,
                        std::uint8_t const & byte,
                        TExtent const & extent)
                    -> mem::view::cuda::detail::TaskSetCuda<
                        TDim,
                        TView,
                        TExtent>
                    {
                        return
                            mem::view::cuda::detail::TaskSetCuda<
                                TDim,
                                TView,
                                TExtent>(
                                    view,
                                    byte,
                                    extent);
                    }
                };
            }
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA non-blocking device queue 1D set enqueue trait specialization.
            template<
                typename TView,
                typename TExtent>
            struct Enqueue<
                queue::QueueCudaRtNonBlocking,
                mem::view::cuda::detail::TaskSetCuda<dim::DimInt<1u>, TView, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtNonBlocking & queue,
                    mem::view::cuda::detail::TaskSetCuda<dim::DimInt<1u>, TView, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TView>::value == 1u,
                        "The destination buffer is required to be 1-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    using Idx = idx::Idx<TExtent>;

                    auto & view(task.m_view);
                    auto const & byte(task.m_byte);
                    auto const & extent(task.m_extent);
                    auto const & iDevice(task.m_iDevice);

                    auto const extentWidth(extent::getWidth(extent));

                    if(extentWidth == 0)
                    {
                        return;
                    }

                    auto const extentWidthBytes(extentWidth * static_cast<Idx>(sizeof(elem::Elem<TView>)));
#if !defined(NDEBUG)
                    auto const dstWidth(extent::getWidth(view));
#endif
                    auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(view)));
                    ALPAKA_ASSERT(extentWidth <= dstWidth);

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemsetAsync(
                            dstNativePtr,
                            static_cast<int>(byte),
                            static_cast<size_t>(extentWidthBytes),
                            queue.m_spQueueImpl->m_CudaQueue));
                }
            };
            //#############################################################################
            //! The CUDA blocking device queue 1D set enqueue trait specialization.
            template<
                typename TView,
                typename TExtent>
            struct Enqueue<
                queue::QueueCudaRtBlocking,
                mem::view::cuda::detail::TaskSetCuda<dim::DimInt<1u>, TView, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtBlocking &,
                    mem::view::cuda::detail::TaskSetCuda<dim::DimInt<1u>, TView, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TView>::value == 1u,
                        "The destination buffer is required to be 1-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    using Idx = idx::Idx<TExtent>;

                    auto & view(task.m_view);
                    auto const & byte(task.m_byte);
                    auto const & extent(task.m_extent);
                    auto const & iDevice(task.m_iDevice);

                    auto const extentWidth(extent::getWidth(extent));

                    if(extentWidth == 0)
                    {
                        return;
                    }

                    auto const extentWidthBytes(extentWidth * static_cast<Idx>(sizeof(elem::Elem<TView>)));
#if !defined(NDEBUG)
                    auto const dstWidth(extent::getWidth(view));
#endif
                    auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(view)));
                    ALPAKA_ASSERT(extentWidth <= dstWidth);

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset(
                            dstNativePtr,
                            static_cast<int>(byte),
                            static_cast<size_t>(extentWidthBytes)));
                }
            };
            //#############################################################################
            //! The CUDA non-blocking device queue 2D set enqueue trait specialization.
            template<
                typename TView,
                typename TExtent>
            struct Enqueue<
                queue::QueueCudaRtNonBlocking,
                mem::view::cuda::detail::TaskSetCuda<dim::DimInt<2u>, TView, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtNonBlocking & queue,
                    mem::view::cuda::detail::TaskSetCuda<dim::DimInt<2u>, TView, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TView>::value == 2u,
                        "The destination buffer is required to be 2-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    using Idx = idx::Idx<TExtent>;

                    auto & view(task.m_view);
                    auto const & byte(task.m_byte);
                    auto const & extent(task.m_extent);
                    auto const & iDevice(task.m_iDevice);

                    auto const extentWidth(extent::getWidth(extent));
                    auto const extentHeight(extent::getHeight(extent));

                    if(extentWidth == 0 || extentHeight == 0)
                    {
                        return;
                    }

                    auto const extentWidthBytes(extentWidth * static_cast<Idx>(sizeof(elem::Elem<TView>)));

#if !defined(NDEBUG)
                    auto const dstWidth(extent::getWidth(view));
                    auto const dstHeight(extent::getHeight(view));
#endif
                    auto const dstPitchBytesX(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(view));
                    auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(view)));
                    ALPAKA_ASSERT(extentWidth <= dstWidth);
                    ALPAKA_ASSERT(extentHeight <= dstHeight);

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset2DAsync(
                            dstNativePtr,
                            static_cast<size_t>(dstPitchBytesX),
                            static_cast<int>(byte),
                            static_cast<size_t>(extentWidthBytes),
                            static_cast<size_t>(extentHeight),
                            queue.m_spQueueImpl->m_CudaQueue));
                }
            };
            //#############################################################################
            //! The CUDA blocking device queue 2D set enqueue trait specialization.
            template<
                typename TView,
                typename TExtent>
            struct Enqueue<
                queue::QueueCudaRtBlocking,
                mem::view::cuda::detail::TaskSetCuda<dim::DimInt<2u>, TView, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtBlocking &,
                    mem::view::cuda::detail::TaskSetCuda<dim::DimInt<2u>, TView, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TView>::value == 2u,
                        "The destination buffer is required to be 2-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    using Idx = idx::Idx<TExtent>;

                    auto & view(task.m_view);
                    auto const & byte(task.m_byte);
                    auto const & extent(task.m_extent);
                    auto const & iDevice(task.m_iDevice);

                    auto const extentWidth(extent::getWidth(extent));
                    auto const extentHeight(extent::getHeight(extent));

                    if(extentWidth == 0 || extentHeight == 0)
                    {
                        return;
                    }

                    auto const extentWidthBytes(extentWidth * static_cast<Idx>(sizeof(elem::Elem<TView>)));

#if !defined(NDEBUG)
                    auto const dstWidth(extent::getWidth(view));
                    auto const dstHeight(extent::getHeight(view));
#endif
                    auto const dstPitchBytesX(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(view));
                    auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(view)));
                    ALPAKA_ASSERT(extentWidth <= dstWidth);
                    ALPAKA_ASSERT(extentHeight <= dstHeight);

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));

                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset2D(
                            dstNativePtr,
                            static_cast<size_t>(dstPitchBytesX),
                            static_cast<int>(byte),
                            static_cast<size_t>(extentWidthBytes),
                            static_cast<size_t>(extentHeight)));
                }
            };
            //#############################################################################
            //! The CUDA non-blocking device queue 3D set enqueue trait specialization.
            template<
                typename TView,
                typename TExtent>
            struct Enqueue<
                queue::QueueCudaRtNonBlocking,
                mem::view::cuda::detail::TaskSetCuda<dim::DimInt<3u>, TView, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtNonBlocking & queue,
                    mem::view::cuda::detail::TaskSetCuda<dim::DimInt<3u>, TView, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TView>::value == 3u,
                        "The destination buffer is required to be 3-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    using Elem = alpaka::elem::Elem<TView>;
                    using Idx = idx::Idx<TExtent>;

                    auto & view(task.m_view);
                    auto const & byte(task.m_byte);
                    auto const & extent(task.m_extent);
                    auto const & iDevice(task.m_iDevice);

                    auto const extentWidth(extent::getWidth(extent));
                    auto const extentHeight(extent::getHeight(extent));
                    auto const extentDepth(extent::getDepth(extent));

                    // This is not only an optimization but also prevents a division by zero.
                    if(extentWidth == 0 || extentHeight == 0 || extentDepth == 0)
                    {
                        return;
                    }

                    auto const dstWidth(extent::getWidth(view));
#if !defined(NDEBUG)
                    auto const dstHeight(extent::getHeight(view));
                    auto const dstDepth(extent::getDepth(view));
#endif
                    auto const dstPitchBytesX(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(view));
                    auto const dstPitchBytesY(mem::view::getPitchBytes<dim::Dim<TView>::value - (2u % dim::Dim<TView>::value)>(view));
                    auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(view)));
                    ALPAKA_ASSERT(extentWidth <= dstWidth);
                    ALPAKA_ASSERT(extentHeight <= dstHeight);
                    ALPAKA_ASSERT(extentDepth <= dstDepth);

                    // Fill CUDA parameter structures.
                    cudaPitchedPtr const cudaPitchedPtrVal(
                        make_cudaPitchedPtr(
                            dstNativePtr,
                            static_cast<size_t>(dstPitchBytesX),
                            static_cast<size_t>(dstWidth * static_cast<Idx>(sizeof(Elem))),
                            static_cast<size_t>(dstPitchBytesY / dstPitchBytesX)));

                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            static_cast<size_t>(extentWidth * static_cast<Idx>(sizeof(Elem))),
                            static_cast<size_t>(extentHeight),
                            static_cast<size_t>(extentDepth)));

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset3DAsync(
                            cudaPitchedPtrVal,
                            static_cast<int>(byte),
                            cudaExtentVal,
                            queue.m_spQueueImpl->m_CudaQueue));
                }
            };
            //#############################################################################
            //! The CUDA blocking device queue 3D set enqueue trait specialization.
            template<
                typename TView,
                typename TExtent>
            struct Enqueue<
                queue::QueueCudaRtBlocking,
                mem::view::cuda::detail::TaskSetCuda<dim::DimInt<3u>, TView, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtBlocking &,
                    mem::view::cuda::detail::TaskSetCuda<dim::DimInt<3u>, TView, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        dim::Dim<TView>::value == 3u,
                        "The destination buffer is required to be 3-dimensional for this specialization!");
                    static_assert(
                        dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    using Elem = alpaka::elem::Elem<TView>;
                    using Idx = idx::Idx<TExtent>;

                    auto & view(task.m_view);
                    auto const & byte(task.m_byte);
                    auto const & extent(task.m_extent);
                    auto const & iDevice(task.m_iDevice);

                    auto const extentWidth(extent::getWidth(extent));
                    auto const extentHeight(extent::getHeight(extent));
                    auto const extentDepth(extent::getDepth(extent));

                    // This is not only an optimization but also prevents a division by zero.
                    if(extentWidth == 0 || extentHeight == 0 || extentDepth == 0)
                    {
                        return;
                    }

                    auto const dstWidth(extent::getWidth(view));
#if !defined(NDEBUG)
                    auto const dstHeight(extent::getHeight(view));
                    auto const dstDepth(extent::getDepth(view));
#endif
                    auto const dstPitchBytesX(mem::view::getPitchBytes<dim::Dim<TView>::value - 1u>(view));
                    auto const dstPitchBytesY(mem::view::getPitchBytes<dim::Dim<TView>::value - (2u % dim::Dim<TView>::value)>(view));
                    auto const dstNativePtr(reinterpret_cast<void *>(mem::view::getPtrNative(view)));
                    ALPAKA_ASSERT(extentWidth <= dstWidth);
                    ALPAKA_ASSERT(extentHeight <= dstHeight);
                    ALPAKA_ASSERT(extentDepth <= dstDepth);

                    // Fill CUDA parameter structures.
                    cudaPitchedPtr const cudaPitchedPtrVal(
                        make_cudaPitchedPtr(
                            dstNativePtr,
                            static_cast<size_t>(dstPitchBytesX),
                            static_cast<size_t>(dstWidth * static_cast<Idx>(sizeof(Elem))),
                            static_cast<size_t>(dstPitchBytesY / dstPitchBytesX)));

                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            static_cast<size_t>(extentWidth * static_cast<Idx>(sizeof(Elem))),
                            static_cast<size_t>(extentHeight),
                            static_cast<size_t>(extentDepth)));

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            iDevice));
                    // Initiate the memory set.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaMemset3D(
                            cudaPitchedPtrVal,
                            static_cast<int>(byte),
                            cudaExtentVal));
                }
            };
        }
    }
}

#endif
