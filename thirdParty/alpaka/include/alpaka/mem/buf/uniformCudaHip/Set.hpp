/* Copyright 2023 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
 *                Antonio Di Pilato, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtBlocking.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/wait/Traits.hpp"

#include <cstddef>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    template<typename TApi>
    class DevUniformCudaHipRt;

    namespace detail
    {
        //! The CUDA/HIP memory set task base.
        template<typename TApi, typename TDim, typename TView, typename TExtent>
        struct TaskSetUniformCudaHipBase
        {
            TaskSetUniformCudaHipBase(TView& view, std::uint8_t const& byte, TExtent const& extent)
                : m_view(view)
                , m_byte(byte)
                , m_extent(extent)
                , m_iDevice(getDev(view).getNativeHandle())
            {
            }

        protected:
            TView& m_view;
            std::uint8_t const m_byte;
            TExtent const m_extent;
            std::int32_t const m_iDevice;
        };

        //! The CUDA/HIP memory set task.
        template<typename TApi, typename TDim, typename TView, typename TExtent>
        struct TaskSetUniformCudaHip;

        //! The scalar CUDA/HIP memory set task.
        template<typename TApi, typename TView, typename TExtent>
        struct TaskSetUniformCudaHip<TApi, DimInt<0u>, TView, TExtent>
            : public TaskSetUniformCudaHipBase<TApi, DimInt<0u>, TView, TExtent>
        {
            template<typename TViewFwd>
            TaskSetUniformCudaHip(TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent)
                : TaskSetUniformCudaHipBase<TApi, DimInt<0u>, TView, TExtent>(
                    std::forward<TViewFwd>(view),
                    byte,
                    extent)
            {
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
                // Initiate the memory set.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memsetAsync(
                    getPtrNative(this->m_view),
                    static_cast<int>(this->m_byte),
                    sizeof(Elem<TView>),
                    queue.getNativeHandle()));
            }
        };

        //! The 1D CUDA/HIP memory set task.
        template<typename TApi, typename TView, typename TExtent>
        struct TaskSetUniformCudaHip<TApi, DimInt<1u>, TView, TExtent>
            : public TaskSetUniformCudaHipBase<TApi, DimInt<1u>, TView, TExtent>
        {
            template<typename TViewFwd>
            TaskSetUniformCudaHip(TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent)
                : TaskSetUniformCudaHipBase<TApi, DimInt<1u>, TView, TExtent>(
                    std::forward<TViewFwd>(view),
                    byte,
                    extent)
            {
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
                auto& view = this->m_view;
                auto const& extent = this->m_extent;

                auto const extentWidth = getWidth(extent);
                ALPAKA_ASSERT(extentWidth <= getWidth(view));

                if(extentWidth == 0)
                {
                    return;
                }

                // Initiate the memory set.
                auto const extentWidthBytes = static_cast<std::size_t>(extentWidth) * sizeof(Elem<TView>);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memsetAsync(
                    getPtrNative(view),
                    static_cast<int>(this->m_byte),
                    extentWidthBytes,
                    queue.getNativeHandle()));
            }
        };

        //! The 2D CUDA/HIP memory set task.
        template<typename TApi, typename TView, typename TExtent>
        struct TaskSetUniformCudaHip<TApi, DimInt<2u>, TView, TExtent>
            : public TaskSetUniformCudaHipBase<TApi, DimInt<2u>, TView, TExtent>
        {
            template<typename TViewFwd>
            TaskSetUniformCudaHip(TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent)
                : TaskSetUniformCudaHipBase<TApi, DimInt<2u>, TView, TExtent>(
                    std::forward<TViewFwd>(view),
                    byte,
                    extent)
            {
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
                auto& view = this->m_view;
                auto const& extent = this->m_extent;

                auto const extentWidth = getWidth(extent);
                auto const extentHeight = getHeight(extent);

                if(extentWidth == 0 || extentHeight == 0)
                {
                    return;
                }

                auto const extentWidthBytes = static_cast<std::size_t>(extentWidth) * sizeof(Elem<TView>);

#    if !defined(NDEBUG)
                auto const dstWidth = getWidth(view);
                auto const dstHeight = getHeight(view);
#    endif
                auto const dstRowPitchBytes = static_cast<std::size_t>(getPitchesInBytes(view)[0]);
                auto const dstNativePtr = reinterpret_cast<void*>(getPtrNative(view));
                ALPAKA_ASSERT(extentWidth <= dstWidth);
                ALPAKA_ASSERT(extentHeight <= dstHeight);

                // Initiate the memory set.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memset2DAsync(
                    dstNativePtr,
                    dstRowPitchBytes,
                    static_cast<int>(this->m_byte),
                    extentWidthBytes,
                    static_cast<std::size_t>(extentHeight),
                    queue.getNativeHandle()));
            }
        };

        //! The 3D CUDA/HIP memory set task.
        template<typename TApi, typename TView, typename TExtent>
        struct TaskSetUniformCudaHip<TApi, DimInt<3u>, TView, TExtent>
            : public TaskSetUniformCudaHipBase<TApi, DimInt<3u>, TView, TExtent>
        {
            template<typename TViewFwd>
            TaskSetUniformCudaHip(TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent)
                : TaskSetUniformCudaHipBase<TApi, DimInt<3u>, TView, TExtent>(
                    std::forward<TViewFwd>(view),
                    byte,
                    extent)
            {
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
                using Elem = alpaka::Elem<TView>;

                auto& view = this->m_view;
                auto const& extent = this->m_extent;

                auto const extentWidth = getWidth(extent);
                auto const extentHeight = getHeight(extent);
                auto const extentDepth = getDepth(extent);

                // This is not only an optimization but also prevents a division by zero.
                if(extentWidth == 0 || extentHeight == 0 || extentDepth == 0)
                {
                    return;
                }

                auto const dstWidth = getWidth(view);
#    if !defined(NDEBUG)
                auto const dstHeight = getHeight(view);
                auto const dstDepth = getDepth(view);
#    endif
                auto const [dstSlicePitchBytes, dstRowPitchBytes, _] = getPitchesInBytes(view);
                auto const dstNativePtr = reinterpret_cast<void*>(getPtrNative(view));
                ALPAKA_ASSERT(extentWidth <= dstWidth);
                ALPAKA_ASSERT(extentHeight <= dstHeight);
                ALPAKA_ASSERT(extentDepth <= dstDepth);

                // Fill CUDA parameter structures.
                typename TApi::PitchedPtr_t const pitchedPtrVal = TApi::makePitchedPtr(
                    dstNativePtr,
                    static_cast<std::size_t>(dstRowPitchBytes),
                    static_cast<std::size_t>(dstWidth) * sizeof(Elem),
                    static_cast<std::size_t>(dstSlicePitchBytes / dstRowPitchBytes));

                typename TApi::Extent_t const extentVal = TApi::makeExtent(
                    static_cast<std::size_t>(extentWidth) * sizeof(Elem),
                    static_cast<std::size_t>(extentHeight),
                    static_cast<std::size_t>(extentDepth));

                // Initiate the memory set.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memset3DAsync(
                    pitchedPtrVal,
                    static_cast<int>(this->m_byte),
                    extentVal,
                    queue.getNativeHandle()));
            }
        };
    } // namespace detail

    namespace trait
    {
        //! The CUDA device memory set trait specialization.
        template<typename TApi, typename TDim>
        struct CreateTaskMemset<TDim, DevUniformCudaHipRt<TApi>>
        {
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(TView& view, std::uint8_t const& byte, TExtent const& extent)
                -> alpaka::detail::TaskSetUniformCudaHip<TApi, TDim, TView, TExtent>
            {
                return alpaka::detail::TaskSetUniformCudaHip<TApi, TDim, TView, TExtent>(view, byte, extent);
            }
        };

        //! The CUDA non-blocking device queue scalar set enqueue trait specialization.
        template<typename TApi, typename TView, typename TExtent>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking<TApi>,
            alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<0u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<0u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The CUDA blocking device queue scalar set enqueue trait specialization.
        template<typename TApi, typename TView, typename TExtent>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking<TApi>,
            alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<0u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<0u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);

                wait(queue);
            }
        };

        //! The CUDA non-blocking device queue 1D set enqueue trait specialization.
        template<typename TApi, typename TView, typename TExtent>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking<TApi>,
            alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<1u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<1u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The CUDA blocking device queue 1D set enqueue trait specialization.
        template<typename TApi, typename TView, typename TExtent>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking<TApi>,
            alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<1u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<1u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);

                wait(queue);
            }
        };

        //! The CUDA non-blocking device queue 2D set enqueue trait specialization.
        template<typename TApi, typename TView, typename TExtent>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking<TApi>,
            alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<2u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<2u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The CUDA blocking device queue 2D set enqueue trait specialization.
        template<typename TApi, typename TView, typename TExtent>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking<TApi>,
            alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<2u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<2u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);

                wait(queue);
            }
        };

        //! The CUDA non-blocking device queue 3D set enqueue trait specialization.
        template<typename TApi, typename TView, typename TExtent>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking<TApi>,
            alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<3u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<3u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The CUDA blocking device queue 3D set enqueue trait specialization.
        template<typename TApi, typename TView, typename TExtent>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking<TApi>,
            alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<3u>, TView, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                alpaka::detail::TaskSetUniformCudaHip<TApi, DimInt<3u>, TView, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                task.enqueue(queue);

                wait(queue);
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
