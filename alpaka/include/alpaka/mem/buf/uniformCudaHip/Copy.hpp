/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 *                Bernhard Manfred Gruber, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtBlocking.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp"

#include <cstddef>
#include <cstdint>
#include <set>
#include <tuple>
#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    namespace detail
    {
        //! The CUDA/HIP memory copy trait.
        template<typename TApi, typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyUniformCudaHip;

        //! The scalar CUDA/HIP memory copy trait.
        template<typename TApi, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyUniformCudaHip<TApi, DimInt<0u>, TViewDst, TViewSrc, TExtent>
        {
            using Idx = alpaka::Idx<TExtent>;

            template<typename TViewDstFwd>
            ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                [[maybe_unused]] TExtent const& extent,
                typename TApi::MemcpyKind_t const& uniformMemCpyKind,
                int const& iDstDevice,
                int const& iSrcDevice)
                : m_uniformMemCpyKind(uniformMemCpyKind)
                , m_iDstDevice(iDstDevice)
                , m_iSrcDevice(iSrcDevice)
                , m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                ALPAKA_ASSERT(getExtentProduct(extent) == 1);
#    endif
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                // cudaMemcpy variants on cudaMallocAsync'ed memory need to be called with the correct device,
                // see https://github.com/fwyzard/nvidia_bug_3446335 .
                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(m_iDstDevice));
                // Initiate the memory copy.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memcpyAsync(
                    m_dstMemNative,
                    m_srcMemNative,
                    sizeof(Elem<TViewDst>),
                    m_uniformMemCpyKind,
                    queue.getNativeHandle()));
            }

        private:
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " ddev: " << m_iDstDevice << " ew: " << Idx(1u)
                          << " ewb: " << static_cast<Idx>(sizeof(Elem<TViewDst>)) << " dw: " << Idx(1u)
                          << " dptr: " << m_dstMemNative << " sdev: " << m_iSrcDevice << " sw: " << Idx(1u)
                          << " sptr: " << m_srcMemNative << std::endl;
            }
#    endif

            typename TApi::MemcpyKind_t m_uniformMemCpyKind;
            int m_iDstDevice;
            int m_iSrcDevice;
            void* m_dstMemNative;
            void const* m_srcMemNative;
        };

        //! The 1D CUDA/HIP memory copy trait.
        template<typename TApi, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyUniformCudaHip<TApi, DimInt<1u>, TViewDst, TViewSrc, TExtent>
        {
            using Idx = alpaka::Idx<TExtent>;

            template<typename TViewDstFwd>
            ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                typename TApi::MemcpyKind_t const& uniformMemCpyKind,
                int const& iDstDevice,
                int const& iSrcDevice)
                : m_uniformMemCpyKind(uniformMemCpyKind)
                , m_iDstDevice(iDstDevice)
                , m_iSrcDevice(iSrcDevice)
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                , m_extentWidth(getWidth(extent))
                , m_dstWidth(static_cast<Idx>(getWidth(viewDst)))
                , m_srcWidth(static_cast<Idx>(getWidth(viewSrc)))
#    endif
                , m_extentWidthBytes(static_cast<std::size_t>(getWidth(extent)) * sizeof(Elem<TViewDst>))
                , m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
#    endif
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                if(m_extentWidthBytes == std::size_t{0})
                {
                    return;
                }

                // cudaMemcpy variants on cudaMallocAsync'ed memory need to be called with the correct device,
                // see https://github.com/fwyzard/nvidia_bug_3446335 .
                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(m_iDstDevice));
                // Initiate the memory copy.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memcpyAsync(
                    m_dstMemNative,
                    m_srcMemNative,
                    m_extentWidthBytes,
                    m_uniformMemCpyKind,
                    queue.getNativeHandle()));
            }

        private:
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " ddev: " << m_iDstDevice << " ew: " << m_extentWidth
                          << " ewb: " << m_extentWidthBytes << " dw: " << m_dstWidth << " dptr: " << m_dstMemNative
                          << " sdev: " << m_iSrcDevice << " sw: " << m_srcWidth << " sptr: " << m_srcMemNative
                          << std::endl;
            }
#    endif

            typename TApi::MemcpyKind_t m_uniformMemCpyKind;
            int m_iDstDevice;
            int m_iSrcDevice;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            Idx m_extentWidth;
            Idx m_dstWidth;
            Idx m_srcWidth;
#    endif
            std::size_t m_extentWidthBytes;
            void* m_dstMemNative;
            void const* m_srcMemNative;
        };

        //! The 2D CUDA/HIP memory copy trait.
        template<typename TApi, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyUniformCudaHip<TApi, DimInt<2u>, TViewDst, TViewSrc, TExtent>
        {
            using Idx = alpaka::Idx<TExtent>;

            template<typename TViewDstFwd>
            ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                typename TApi::MemcpyKind_t const& uniformMemcpyKind,
                int const& iDstDevice,
                int const& iSrcDevice)
                : m_uniformMemCpyKind(uniformMemcpyKind)
                , m_iDstDevice(iDstDevice)
                , m_iSrcDevice(iSrcDevice)
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                , m_extentWidth(getWidth(extent))
#    endif
                , m_extentWidthBytes(static_cast<std::size_t>(getWidth(extent)) * sizeof(Elem<TViewDst>))
                , m_dstWidth(static_cast<Idx>(getWidth(viewDst)))
                , m_srcWidth(static_cast<Idx>(getWidth(viewSrc)))
                , m_extentHeight(getHeight(extent))
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                , m_dstHeight(static_cast<Idx>(getHeight(viewDst)))
                , m_srcHeight(static_cast<Idx>(getHeight(viewSrc)))
#    endif
                , m_dstRowPitchBytes(static_cast<std::size_t>(getPitchesInBytes(viewDst)[0]))
                , m_srcRowPitchBytes(static_cast<std::size_t>(getPitchesInBytes(viewSrc)[0]))
                , m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                ALPAKA_ASSERT(m_extentHeight <= m_dstHeight);
                ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
                ALPAKA_ASSERT(m_extentHeight <= m_srcHeight);
                ALPAKA_ASSERT(m_extentWidthBytes <= m_dstRowPitchBytes);
                ALPAKA_ASSERT(m_extentWidthBytes <= m_srcRowPitchBytes);
#    endif
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                // This is not only an optimization but also prevents a division by zero.
                if(m_extentWidthBytes == std::size_t{0} || m_extentHeight == 0)
                {
                    return;
                }

                // cudaMemcpy variants on cudaMallocAsync'ed memory need to be called with the correct device,
                // see https://github.com/fwyzard/nvidia_bug_3446335 .
                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(m_iDstDevice));
                // Initiate the memory copy.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memcpy2DAsync(
                    m_dstMemNative,
                    m_dstRowPitchBytes,
                    m_srcMemNative,
                    m_srcRowPitchBytes,
                    m_extentWidthBytes,
                    static_cast<std::size_t>(m_extentHeight),
                    m_uniformMemCpyKind,
                    queue.getNativeHandle()));
            }

        private:
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " ew: " << m_extentWidth << " eh: " << m_extentHeight
                          << " ewb: " << m_extentWidthBytes << " ddev: " << m_iDstDevice << " dw: " << m_dstWidth
                          << " dh: " << m_dstHeight << " dptr: " << m_dstMemNative << " dpitch: " << m_dstRowPitchBytes
                          << " sdev: " << m_iSrcDevice << " sw: " << m_srcWidth << " sh: " << m_srcHeight
                          << " sptr: " << m_srcMemNative << " spitch: " << m_srcRowPitchBytes << std::endl;
            }
#    endif

            typename TApi::MemcpyKind_t m_uniformMemCpyKind;
            int m_iDstDevice;
            int m_iSrcDevice;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            Idx m_extentWidth;
#    endif
            std::size_t m_extentWidthBytes;
            Idx m_dstWidth;
            Idx m_srcWidth;

            Idx m_extentHeight;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            Idx m_dstHeight;
            Idx m_srcHeight;
#    endif
            std::size_t m_dstRowPitchBytes;
            std::size_t m_srcRowPitchBytes;

            void* m_dstMemNative;
            void const* m_srcMemNative;
        };

        //! The 3D CUDA/HIP memory copy trait.
        template<typename TApi, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyUniformCudaHip<TApi, DimInt<3u>, TViewDst, TViewSrc, TExtent>
        {
            using Idx = alpaka::Idx<TExtent>;

            template<typename TViewDstFwd>
            ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                typename TApi::MemcpyKind_t const& uniformMemcpyKind,
                int const& iDstDevice,
                int const& iSrcDevice)
                : m_uniformMemCpyKind(uniformMemcpyKind)
                , m_iDstDevice(iDstDevice)
                , m_iSrcDevice(iSrcDevice)
                , m_extentWidth(getWidth(extent))
                , m_extentWidthBytes(static_cast<std::size_t>(m_extentWidth) * sizeof(Elem<TViewDst>))
                , m_dstWidth(static_cast<Idx>(getWidth(viewDst)))
                , m_srcWidth(static_cast<Idx>(getWidth(viewSrc)))
                , m_extentHeight(getHeight(extent))
                , m_extentDepth(getDepth(extent))
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                , m_dstHeight(static_cast<Idx>(getHeight(viewDst)))
                , m_srcHeight(static_cast<Idx>(getHeight(viewSrc)))
                , m_dstDepth(static_cast<Idx>(getDepth(viewDst)))
                , m_srcDepth(static_cast<Idx>(getDepth(viewSrc)))
#    endif
                , m_dstRowPitchBytes(static_cast<std::size_t>(getPitchesInBytes(viewDst)[1]))
                , m_srcRowPitchBytes(static_cast<std::size_t>(getPitchesInBytes(viewSrc)[1]))
                , m_dstSlicePitchBytes(static_cast<std::size_t>(getPitchesInBytes(viewDst)[0]))
                , m_srcSlicePitchBytes(static_cast<std::size_t>(getPitchesInBytes(viewSrc)[0]))
                , m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                ALPAKA_ASSERT(m_extentHeight <= m_dstHeight);
                ALPAKA_ASSERT(m_extentDepth <= m_dstDepth);
                ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
                ALPAKA_ASSERT(m_extentHeight <= m_srcHeight);
                ALPAKA_ASSERT(m_extentDepth <= m_srcDepth);
                ALPAKA_ASSERT(m_extentWidthBytes <= m_dstRowPitchBytes);
                ALPAKA_ASSERT(m_extentWidthBytes <= m_srcRowPitchBytes);
#    endif
            }

            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                // This is not only an optimization but also prevents a division by zero.
                if(m_extentWidthBytes == std::size_t{0} || m_extentHeight == 0 || m_extentDepth == 0)
                {
                    return;
                }

                // Create the struct describing the copy.
                typename TApi::Memcpy3DParms_t const uniformCudaHipMemCpy3DParms(buildUniformCudaHipMemcpy3DParms());

                // cudaMemcpy variants on cudaMallocAsync'ed memory need to be called with the correct device,
                // see https://github.com/fwyzard/nvidia_bug_3446335 .
                // Set the current device.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(m_iDstDevice));

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    TApi::memcpy3DAsync(&uniformCudaHipMemCpy3DParms, queue.getNativeHandle()));
            }

        private:
            ALPAKA_FN_HOST auto buildUniformCudaHipMemcpy3DParms() const -> typename TApi::Memcpy3DParms_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                // Fill CUDA/HIP parameter structure.
                typename TApi::Memcpy3DParms_t memCpy3DParms{}; // zero-init required per CUDA documentation
                memCpy3DParms.srcPtr = TApi::makePitchedPtr(
                    const_cast<void*>(m_srcMemNative),
                    m_srcRowPitchBytes,
                    static_cast<std::size_t>(m_srcWidth),
                    m_srcSlicePitchBytes / m_srcRowPitchBytes);
                memCpy3DParms.dstPtr = TApi::makePitchedPtr(
                    m_dstMemNative,
                    m_dstRowPitchBytes,
                    static_cast<std::size_t>(m_dstWidth),
                    m_dstSlicePitchBytes / m_dstRowPitchBytes);
                memCpy3DParms.extent = TApi::makeExtent(
                    m_extentWidthBytes,
                    static_cast<std::size_t>(m_extentHeight),
                    static_cast<std::size_t>(m_extentDepth));
                memCpy3DParms.kind = m_uniformMemCpyKind;
                return memCpy3DParms;
            }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " ew: " << m_extentWidth << " eh: " << m_extentHeight
                          << " ed: " << m_extentDepth << " ewb: " << m_extentWidthBytes << " ddev: " << m_iDstDevice
                          << " dw: " << m_dstWidth << " dh: " << m_dstHeight << " dd: " << m_dstDepth
                          << " dptr: " << m_dstMemNative << " drowpitch: " << m_dstRowPitchBytes
                          << " dslicepitch: " << m_dstSlicePitchBytes << " sdev: " << m_iSrcDevice
                          << " sw: " << m_srcWidth << " sh: " << m_srcHeight << " sd: " << m_srcDepth
                          << " sptr: " << m_srcMemNative << " srowpitch: " << m_srcRowPitchBytes
                          << " sslicepitch: " << m_srcSlicePitchBytes << std::endl;
            }
#    endif
            typename TApi::MemcpyKind_t m_uniformMemCpyKind;
            int m_iDstDevice;
            int m_iSrcDevice;

            Idx m_extentWidth;
            std::size_t m_extentWidthBytes;
            Idx m_dstWidth;
            Idx m_srcWidth;

            Idx m_extentHeight;
            Idx m_extentDepth;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            Idx m_dstHeight;
            Idx m_srcHeight;
            Idx m_dstDepth;
            Idx m_srcDepth;
#    endif
            std::size_t m_dstRowPitchBytes;
            std::size_t m_srcRowPitchBytes;
            std::size_t m_dstSlicePitchBytes;
            std::size_t m_srcSlicePitchBytes;

            void* m_dstMemNative;
            void const* m_srcMemNative;
        };
    } // namespace detail

    // Trait specializations for CreateTaskMemcpy.
    namespace trait
    {
        //! The CUDA/HIP to CPU memory copy trait specialization.
        template<typename TApi, typename TDim>
        struct CreateTaskMemcpy<TDim, DevCpu, DevUniformCudaHipRt<TApi>>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::
                TaskCopyUniformCudaHip<TApi, TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto const iDevice = getDev(viewSrc).getNativeHandle();

                return {
                    std::forward<TViewDstFwd>(viewDst),
                    viewSrc,
                    extent,
                    TApi::memcpyDeviceToHost,
                    iDevice,
                    iDevice};
            }
        };

        //! The CPU to CUDA/HIP memory copy trait specialization.
        template<typename TApi, typename TDim>
        struct CreateTaskMemcpy<TDim, DevUniformCudaHipRt<TApi>, DevCpu>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::
                TaskCopyUniformCudaHip<TApi, TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto const iDevice = getDev(viewDst).getNativeHandle();

                return {
                    std::forward<TViewDstFwd>(viewDst),
                    viewSrc,
                    extent,
                    TApi::memcpyHostToDevice,
                    iDevice,
                    iDevice};
            }
        };

        //! The CUDA/HIP to CUDA/HIP memory copy trait specialization.
        template<typename TApi, typename TDim>
        struct CreateTaskMemcpy<TDim, DevUniformCudaHipRt<TApi>, DevUniformCudaHipRt<TApi>>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::
                TaskCopyUniformCudaHip<TApi, TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto const iDstDevice = getDev(viewDst).getNativeHandle();

                return {
                    std::forward<TViewDstFwd>(viewDst),
                    viewSrc,
                    extent,
                    TApi::memcpyDeviceToDevice,
                    iDstDevice,
                    getDev(viewSrc).getNativeHandle()};
            }
        };

        //! The CUDA/HIP non-blocking device queue scalar copy enqueue trait specialization.
        template<typename TApi, typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking<TApi>,
            alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<0u>, TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<0u>, TViewDst, TViewSrc, TExtent> const& task)
                -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The CUDA/HIP blocking device queue scalar copy enqueue trait specialization.
        template<typename TApi, typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking<TApi>,
            alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<0u>, TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<0u>, TViewDst, TViewSrc, TExtent> const& task)
                -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP non-blocking device queue 1D copy enqueue trait specialization.
        template<typename TApi, typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking<TApi>,
            alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<1u>, TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<1u>, TViewDst, TViewSrc, TExtent> const& task)
                -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The CUDA/HIP blocking device queue 1D copy enqueue trait specialization.
        template<typename TApi, typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking<TApi>,
            alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<1u>, TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<1u>, TViewDst, TViewSrc, TExtent> const& task)
                -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP non-blocking device queue 2D copy enqueue trait specialization.
        template<typename TApi, typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking<TApi>,
            alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<2u>, TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<2u>, TViewDst, TViewSrc, TExtent> const& task)
                -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The CUDA/HIP blocking device queue 2D copy enqueue trait specialization.
        template<typename TApi, typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking<TApi>,
            alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<2u>, TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<2u>, TViewDst, TViewSrc, TExtent> const& task)
                -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(queue.getNativeHandle()));
            }
        };

        //! The CUDA/HIP non-blocking device queue 3D copy enqueue trait specialization.
        template<typename TApi, typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking<TApi>,
            alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<3u>, TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking<TApi>& queue,
                alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<3u>, TViewDst, TViewSrc, TExtent> const& task)
                -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };

        //! The CUDA/HIP blocking device queue 3D copy enqueue trait specialization.
        template<typename TApi, typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking<TApi>,
            alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<3u>, TViewDst, TViewSrc, TExtent>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking<TApi>& queue,
                alpaka::detail::TaskCopyUniformCudaHip<TApi, DimInt<3u>, TViewDst, TViewSrc, TExtent> const& task)
                -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(queue.getNativeHandle()));
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
