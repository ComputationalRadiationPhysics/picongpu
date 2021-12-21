/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner,
 *                Rene Widera
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

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/mem/view/Traits.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>
// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <cstdint>
#    include <set>
#    include <tuple>
#    include <type_traits>

namespace alpaka
{
    namespace detail
    {
        //-----------------------------------------------------------------------------
        //! Not being able to enable peer access does not prevent such device to device memory copies.
        //! However, those copies may be slower because the memory is copied via the CPU.
        inline auto enablePeerAccessIfPossible(const int& devSrc, const int& devDst) -> void
        {
            ALPAKA_ASSERT(devSrc != devDst);

#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wexit-time-destructors"
#    endif
            static std::set<std::pair<int, int>> alreadyCheckedPeerAccessDevices;
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
            auto const devicePair = std::make_pair(devSrc, devDst);

            if(alreadyCheckedPeerAccessDevices.find(devicePair) == alreadyCheckedPeerAccessDevices.end())
            {
                alreadyCheckedPeerAccessDevices.insert(devicePair);

                int canAccessPeer = 0;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(DeviceCanAccessPeer)(&canAccessPeer, devSrc, devDst));

                if(!canAccessPeer)
                {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << __func__ << " Direct peer access between given GPUs is not possible!"
                              << " src=" << devSrc << " dst=" << devDst << std::endl;
#    endif
                    return;
                }
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(devSrc));

                // NOTE: "until access is explicitly disabled using cudaDeviceDisablePeerAccess() or either device is
                // reset using cudaDeviceReset()." We do not remove a device from the enabled device pairs on
                // cudaDeviceReset. Note that access granted by this call is unidirectional and that in order to access
                // memory on the current device from peerDevice, a separate symmetric call to
                // cudaDeviceEnablePeerAccess() is required.
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(DeviceEnablePeerAccess)(devDst, 0));
            }
        }

        //#############################################################################
        //! The CUDA/HIP memory copy trait.
        template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyUniformCudaHip;

        //#############################################################################
        //! The 1D CUDA/HIP memory copy trait.
        template<typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyUniformCudaHip<DimInt<1>, TViewDst, TViewSrc, TExtent>
        {
            using MemcpyKind = ALPAKA_API_PREFIX(MemcpyKind);

            static_assert(!std::is_const<TViewDst>::value, "The destination view can not be const!");

            static_assert(
                Dim<TViewDst>::value == Dim<TViewSrc>::value,
                "The source and the destination view are required to have the same dimensionality!");
            static_assert(
                Dim<TViewDst>::value == Dim<TExtent>::value,
                "The views and the extent are required to have the same dimensionality!");
            // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
            static_assert(
                std::is_same<Elem<TViewDst>, std::remove_const_t<Elem<TViewSrc>>>::value,
                "The source and the destination view are required to have the same element type!");

            using Idx = Idx<TExtent>;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                MemcpyKind const& uniformMemCpyKind,
                int const& iDstDevice,
                int const& iSrcDevice)
                : m_uniformMemCpyKind(uniformMemCpyKind)
                , m_iDstDevice(iDstDevice)
                , m_iSrcDevice(iSrcDevice)
                ,
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                m_extentWidth(extent::getWidth(extent))
                , m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst)))
                , m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc)))
                ,
#    endif
                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Idx>(sizeof(Elem<TViewDst>)))
                , m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
#    endif
            }

            //-----------------------------------------------------------------------------
            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                if(m_extentWidthBytes == 0)
                {
                    return;
                }

                auto const& uniformCudaHipMemCpyKind(m_uniformMemCpyKind);

                if(m_iDstDevice == m_iSrcDevice)
                {
                    // Set the current device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_iDstDevice));
                    // Initiate the memory copy.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(MemcpyAsync)(
                        m_dstMemNative,
                        m_srcMemNative,
                        static_cast<std::size_t>(m_extentWidthBytes),
                        uniformCudaHipMemCpyKind,
                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
                }
                else
                {
                    alpaka::detail::enablePeerAccessIfPossible(m_iSrcDevice, m_iDstDevice);

                    // Initiate the memory copy.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(MemcpyPeerAsync)(
                        m_dstMemNative,
                        m_iDstDevice,
                        m_srcMemNative,
                        m_iSrcDevice,
                        static_cast<std::size_t>(m_extentWidthBytes),
                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
                }
            }

        private:
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " ddev: " << m_iDstDevice << " ew: " << m_extentWidth
                          << " ewb: " << m_extentWidthBytes << " dw: " << m_dstWidth << " dptr: " << m_dstMemNative
                          << " sdev: " << m_iSrcDevice << " sw: " << m_srcWidth << " sptr: " << m_srcMemNative
                          << std::endl;
            }
#    endif

            MemcpyKind m_uniformMemCpyKind;
            int m_iDstDevice;
            int m_iSrcDevice;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            Idx m_extentWidth;
            Idx m_dstWidth;
            Idx m_srcWidth;
#    endif
            Idx m_extentWidthBytes;
            void* m_dstMemNative;
            void const* m_srcMemNative;
        };
        //#############################################################################
        //! The 2D CUDA/HIP memory copy trait.
        template<typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyUniformCudaHip<DimInt<2>, TViewDst, TViewSrc, TExtent>
        {
            using MemcpyKind = ALPAKA_API_PREFIX(MemcpyKind);

            static_assert(!std::is_const<TViewDst>::value, "The destination view can not be const!");

            static_assert(
                Dim<TViewDst>::value == Dim<TViewSrc>::value,
                "The source and the destination view are required to have the same dimensionality!");
            static_assert(
                Dim<TViewDst>::value == Dim<TExtent>::value,
                "The views and the extent are required to have the same dimensionality!");
            // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
            static_assert(
                std::is_same<Elem<TViewDst>, std::remove_const_t<Elem<TViewSrc>>>::value,
                "The source and the destination view are required to have the same element type!");

            using Idx = Idx<TExtent>;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                MemcpyKind const& uniformMemcpyKind,
                int const& iDstDevice,
                int const& iSrcDevice)
                : m_uniformMemCpyKind(uniformMemcpyKind)
                , m_iDstDevice(iDstDevice)
                , m_iSrcDevice(iSrcDevice)
                ,
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                m_extentWidth(extent::getWidth(extent))
                ,
#    endif
                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Idx>(sizeof(Elem<TViewDst>)))
                , m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst)))
                , // required for 3D peer copy
                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc)))
                , // required for 3D peer copy

                m_extentHeight(extent::getHeight(extent))
                , m_dstHeight(static_cast<Idx>(extent::getHeight(viewDst)))
                , // required for 3D peer copy
                m_srcHeight(static_cast<Idx>(extent::getHeight(viewSrc)))
                , // required for 3D peer copy

                m_dstpitchBytesX(static_cast<Idx>(getPitchBytes<Dim<TViewDst>::value - 1u>(viewDst)))
                , m_srcpitchBytesX(static_cast<Idx>(getPitchBytes<Dim<TViewSrc>::value - 1u>(viewSrc)))
                , m_dstPitchBytesY(
                      static_cast<Idx>(getPitchBytes<Dim<TViewDst>::value - (2u % Dim<TViewDst>::value)>(viewDst)))
                , m_srcPitchBytesY(
                      static_cast<Idx>(getPitchBytes<Dim<TViewSrc>::value - (2u % Dim<TViewDst>::value)>(viewSrc)))
                ,

                m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                ALPAKA_ASSERT(m_extentHeight <= m_dstHeight);
                ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
                ALPAKA_ASSERT(m_extentHeight <= m_srcHeight);
                ALPAKA_ASSERT(m_extentWidthBytes <= m_dstpitchBytesX);
                ALPAKA_ASSERT(m_extentWidthBytes <= m_srcpitchBytesX);
#    endif
            }

            //-----------------------------------------------------------------------------
            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                // This is not only an optimization but also prevents a division by zero.
                if(m_extentWidthBytes == 0 || m_extentHeight == 0)
                {
                    return;
                }

                if(m_iDstDevice == m_iSrcDevice)
                {
                    // Set the current device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_iDstDevice));
                    // Initiate the memory copy.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(Memcpy2DAsync)(
                        m_dstMemNative,
                        static_cast<std::size_t>(m_dstpitchBytesX),
                        m_srcMemNative,
                        static_cast<std::size_t>(m_srcpitchBytesX),
                        static_cast<std::size_t>(m_extentWidthBytes),
                        static_cast<std::size_t>(m_extentHeight),
                        m_uniformMemCpyKind,
                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
                }
                else
                {
                    alpaka::detail::enablePeerAccessIfPossible(m_iSrcDevice, m_iDstDevice);
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    // There is no cudaMemcpy2DPeerAsync, therefore we use cudaMemcpy3DPeerAsync.
                    // Create the struct describing the copy.
                    ALPAKA_API_PREFIX(Memcpy3DPeerParms) const memCpy3DPeerParms(buildCudaMemcpy3DPeerParms());
                    // Initiate the memory copy.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        cudaMemcpy3DPeerAsync(&memCpy3DPeerParms, queue.m_spQueueImpl->m_UniformCudaHipQueue));
#    endif
                }
            }

        private:
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto buildCudaMemcpy3DPeerParms() const -> cudaMemcpy3DPeerParms
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                // Fill CUDA parameter structure.
                cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms;
                cudaMemCpy3DPeerParms.dstArray = nullptr; // Either dstArray or dstPtr.
                cudaMemCpy3DPeerParms.dstDevice = m_iDstDevice;
                cudaMemCpy3DPeerParms.dstPos = make_cudaPos(0, 0, 0); // Optional. Offset in bytes.
                cudaMemCpy3DPeerParms.dstPtr = make_cudaPitchedPtr(
                    m_dstMemNative,
                    static_cast<std::size_t>(m_dstpitchBytesX),
                    static_cast<std::size_t>(m_dstWidth),
                    static_cast<std::size_t>(m_dstPitchBytesY / m_dstpitchBytesX));
                cudaMemCpy3DPeerParms.extent = make_cudaExtent(
                    static_cast<std::size_t>(m_extentWidthBytes),
                    static_cast<std::size_t>(m_extentHeight),
                    static_cast<std::size_t>(1u));
                cudaMemCpy3DPeerParms.srcArray = nullptr; // Either srcArray or srcPtr.
                cudaMemCpy3DPeerParms.srcDevice = m_iSrcDevice;
                cudaMemCpy3DPeerParms.srcPos = make_cudaPos(0, 0, 0); // Optional. Offset in bytes.
                cudaMemCpy3DPeerParms.srcPtr = make_cudaPitchedPtr(
                    const_cast<void*>(m_srcMemNative),
                    static_cast<std::size_t>(m_srcpitchBytesX),
                    static_cast<std::size_t>(m_srcWidth),
                    static_cast<std::size_t>(m_srcPitchBytesY / m_srcpitchBytesX));

                return cudaMemCpy3DPeerParms;
            }
#    endif
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " ew: " << m_extentWidth << " eh: " << m_extentHeight
                          << " ewb: " << m_extentWidthBytes << " ddev: " << m_iDstDevice << " dw: " << m_dstWidth
                          << " dh: " << m_dstHeight << " dptr: " << m_dstMemNative << " dpitchb: " << m_dstpitchBytesX
                          << " sdev: " << m_iSrcDevice << " sw: " << m_srcWidth << " sh: " << m_srcHeight
                          << " sptr: " << m_srcMemNative << " spitchb: " << m_srcpitchBytesX << std::endl;
            }
#    endif

            MemcpyKind m_uniformMemCpyKind;
            int m_iDstDevice;
            int m_iSrcDevice;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            Idx m_extentWidth;
#    endif
            Idx m_extentWidthBytes;
            Idx m_dstWidth; // required for 3D peer copy
            Idx m_srcWidth; // required for 3D peer copy

            Idx m_extentHeight;
            Idx m_dstHeight; // required for 3D peer copy
            Idx m_srcHeight; // required for 3D peer copy

            Idx m_dstpitchBytesX;
            Idx m_srcpitchBytesX;
            Idx m_dstPitchBytesY;
            Idx m_srcPitchBytesY;


            void* m_dstMemNative;
            void const* m_srcMemNative;
        };
        //#############################################################################
        //! The 3D CUDA/HIP memory copy trait.
        template<typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyUniformCudaHip<DimInt<3>, TViewDst, TViewSrc, TExtent>
        {
            using MemcpyKind = ALPAKA_API_PREFIX(MemcpyKind);

            static_assert(!std::is_const<TViewDst>::value, "The destination view can not be const!");

            static_assert(
                Dim<TViewDst>::value == Dim<TViewSrc>::value,
                "The source and the destination view are required to have the same dimensionality!");
            static_assert(
                Dim<TViewDst>::value == Dim<TExtent>::value,
                "The views and the extent are required to have the same dimensionality!");
            // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
            static_assert(
                std::is_same<Elem<TViewDst>, std::remove_const_t<Elem<TViewSrc>>>::value,
                "The source and the destination view are required to have the same element type!");

            using Idx = Idx<TExtent>;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                MemcpyKind const& uniformMemcpyKind,
                int const& iDstDevice,
                int const& iSrcDevice)
                : m_uniformMemCpyKind(uniformMemcpyKind)
                , m_iDstDevice(iDstDevice)
                , m_iSrcDevice(iSrcDevice)
                ,

                m_extentWidth(extent::getWidth(extent))
                , m_extentWidthBytes(m_extentWidth * static_cast<Idx>(sizeof(Elem<TViewDst>)))
                , m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst)))
                , m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc)))
                ,

                m_extentHeight(extent::getHeight(extent))
                , m_dstHeight(static_cast<Idx>(extent::getHeight(viewDst)))
                , m_srcHeight(static_cast<Idx>(extent::getHeight(viewSrc)))
                ,

                m_extentDepth(extent::getDepth(extent))
                ,
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                m_dstDepth(static_cast<Idx>(extent::getDepth(viewDst)))
                , m_srcDepth(static_cast<Idx>(extent::getDepth(viewSrc)))
                ,
#    endif
                m_dstpitchBytesX(static_cast<Idx>(getPitchBytes<Dim<TViewDst>::value - 1u>(viewDst)))
                , m_srcpitchBytesX(static_cast<Idx>(getPitchBytes<Dim<TViewSrc>::value - 1u>(viewSrc)))
                , m_dstPitchBytesY(
                      static_cast<Idx>(getPitchBytes<Dim<TViewDst>::value - (2u % Dim<TViewDst>::value)>(viewDst)))
                , m_srcPitchBytesY(
                      static_cast<Idx>(getPitchBytes<Dim<TViewSrc>::value - (2u % Dim<TViewDst>::value)>(viewSrc)))
                ,


                m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                ALPAKA_ASSERT(m_extentHeight <= m_dstHeight);
                ALPAKA_ASSERT(m_extentDepth <= m_dstDepth);
                ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
                ALPAKA_ASSERT(m_extentHeight <= m_srcHeight);
                ALPAKA_ASSERT(m_extentDepth <= m_srcDepth);
                ALPAKA_ASSERT(m_extentWidthBytes <= m_dstpitchBytesX);
                ALPAKA_ASSERT(m_extentWidthBytes <= m_srcpitchBytesX);
#    endif
            }

            //-----------------------------------------------------------------------------
            template<typename TQueue>
            auto enqueue(TQueue& queue) const -> void
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                // This is not only an optimization but also prevents a division by zero.
                if(m_extentWidthBytes == 0 || m_extentHeight == 0 || m_extentDepth == 0)
                {
                    return;
                }

                if(m_iDstDevice == m_iSrcDevice)
                {
                    // Create the struct describing the copy.
                    ALPAKA_API_PREFIX(Memcpy3DParms)
                    const uniformCudaHipMemCpy3DParms(buildUniformCudaHipMemcpy3DParms());
                    // Set the current device.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(m_iDstDevice));

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(
                        Memcpy3DAsync)(&uniformCudaHipMemCpy3DParms, queue.m_spQueueImpl->m_UniformCudaHipQueue));
                }
                else
                {
                    alpaka::detail::enablePeerAccessIfPossible(m_iSrcDevice, m_iDstDevice);
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    // Create the struct describing the copy.
                    cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(buildCudaMemcpy3DPeerParms());
                    // Initiate the memory copy.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        cudaMemcpy3DPeerAsync(&cudaMemCpy3DPeerParms, queue.m_spQueueImpl->m_UniformCudaHipQueue));
#    endif
                }
            }

        private:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto buildUniformCudaHipMemcpy3DParms() const -> ALPAKA_API_PREFIX(Memcpy3DParms)
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                // Fill CUDA/HIP parameter structure.
                ALPAKA_API_PREFIX(Memcpy3DParms) memCpy3DParms;
                memCpy3DParms.srcArray = nullptr; // Either srcArray or srcPtr.
                memCpy3DParms.srcPos
                    = ALPAKA_PP_CONCAT(make_, ALPAKA_API_PREFIX(Pos))(0, 0, 0); // Optional. Offset in bytes.
                memCpy3DParms.srcPtr = ALPAKA_PP_CONCAT(make_, ALPAKA_API_PREFIX(PitchedPtr))(
                    const_cast<void*>(m_srcMemNative),
                    static_cast<std::size_t>(m_srcpitchBytesX),
                    static_cast<std::size_t>(m_srcWidth),
                    static_cast<std::size_t>(m_srcPitchBytesY / m_srcpitchBytesX));
                memCpy3DParms.dstArray = nullptr; // Either dstArray or dstPtr.
                memCpy3DParms.dstPos
                    = ALPAKA_PP_CONCAT(make_, ALPAKA_API_PREFIX(Pos))(0, 0, 0); // Optional. Offset in bytes.
                memCpy3DParms.dstPtr = ALPAKA_PP_CONCAT(make_, ALPAKA_API_PREFIX(PitchedPtr))(
                    m_dstMemNative,
                    static_cast<std::size_t>(m_dstpitchBytesX),
                    static_cast<std::size_t>(m_dstWidth),
                    static_cast<std::size_t>(m_dstPitchBytesY / m_dstpitchBytesX));
                memCpy3DParms.extent = ALPAKA_PP_CONCAT(make_, ALPAKA_API_PREFIX(Extent))(
                    static_cast<std::size_t>(m_extentWidthBytes),
                    static_cast<std::size_t>(m_extentHeight),
                    static_cast<std::size_t>(m_extentDepth));
#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_PLATFORM_NVCC__)
                memCpy3DParms.kind = hipMemcpyKindToCudaMemcpyKind(m_uniformMemCpyKind);
#    else
                memCpy3DParms.kind = m_uniformMemCpyKind;
#    endif
                return memCpy3DParms;
            }

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto buildCudaMemcpy3DPeerParms() const -> cudaMemcpy3DPeerParms
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                // Fill CUDA parameter structure.
                cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms;
                cudaMemCpy3DPeerParms.dstArray = nullptr; // Either dstArray or dstPtr.
                cudaMemCpy3DPeerParms.dstDevice = m_iDstDevice;
                cudaMemCpy3DPeerParms.dstPos = make_cudaPos(0, 0, 0); // Optional. Offset in bytes.
                cudaMemCpy3DPeerParms.dstPtr = make_cudaPitchedPtr(
                    m_dstMemNative,
                    static_cast<std::size_t>(m_dstpitchBytesX),
                    static_cast<std::size_t>(m_dstWidth),
                    static_cast<std::size_t>(m_dstPitchBytesY / m_dstpitchBytesX));
                cudaMemCpy3DPeerParms.extent = make_cudaExtent(
                    static_cast<std::size_t>(m_extentWidthBytes),
                    static_cast<std::size_t>(m_extentHeight),
                    static_cast<std::size_t>(m_extentDepth));
                cudaMemCpy3DPeerParms.srcArray = nullptr; // Either srcArray or srcPtr.
                cudaMemCpy3DPeerParms.srcDevice = m_iSrcDevice;
                cudaMemCpy3DPeerParms.srcPos = make_cudaPos(0, 0, 0); // Optional. Offset in bytes.
                cudaMemCpy3DPeerParms.srcPtr = make_cudaPitchedPtr(
                    const_cast<void*>(m_srcMemNative),
                    static_cast<std::size_t>(m_srcpitchBytesX),
                    static_cast<std::size_t>(m_srcWidth),
                    static_cast<std::size_t>(m_srcPitchBytesY / m_srcpitchBytesX));

                return cudaMemCpy3DPeerParms;
            }
#    endif
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " ew: " << m_extentWidth << " eh: " << m_extentHeight
                          << " ed: " << m_extentDepth << " ewb: " << m_extentWidthBytes << " ddev: " << m_iDstDevice
                          << " dw: " << m_dstWidth << " dh: " << m_dstHeight << " dd: " << m_dstDepth
                          << " dptr: " << m_dstMemNative << " dpitchb: " << m_dstpitchBytesX
                          << " sdev: " << m_iSrcDevice << " sw: " << m_srcWidth << " sh: " << m_srcHeight
                          << " sd: " << m_srcDepth << " sptr: " << m_srcMemNative << " spitchb: " << m_srcpitchBytesX
                          << std::endl;
            }
#    endif
            MemcpyKind m_uniformMemCpyKind;
            int m_iDstDevice;
            int m_iSrcDevice;

            Idx m_extentWidth;
            Idx m_extentWidthBytes;
            Idx m_dstWidth;
            Idx m_srcWidth;

            Idx m_extentHeight;
            Idx m_dstHeight;
            Idx m_srcHeight;

            Idx m_extentDepth;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            Idx m_dstDepth;
            Idx m_srcDepth;
#    endif
            Idx m_dstpitchBytesX;
            Idx m_srcpitchBytesX;
            Idx m_dstPitchBytesY;
            Idx m_srcPitchBytesY;

            void* m_dstMemNative;
            void const* m_srcMemNative;
        };
    } // namespace detail

    //-----------------------------------------------------------------------------
    // Trait specializations for CreateTaskMemcpy.
    namespace traits
    {
        //#############################################################################
        //! The CUDA/HIP to CPU memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevCpu, DevUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::TaskCopyUniformCudaHip<TDim, TViewDst, TViewSrc, TExtent>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto const iDevice(getDev(viewSrc).m_iDevice);

                return alpaka::detail::TaskCopyUniformCudaHip<TDim, TViewDst, TViewSrc, TExtent>(
                    viewDst,
                    viewSrc,
                    extent,
                    ALPAKA_API_PREFIX(MemcpyDeviceToHost),
                    iDevice,
                    iDevice);
            }
        };
        //#############################################################################
        //! The CPU to CUDA/HIP memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevUniformCudaHipRt, DevCpu>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::TaskCopyUniformCudaHip<TDim, TViewDst, TViewSrc, TExtent>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto const iDevice(getDev(viewDst).m_iDevice);

                return alpaka::detail::TaskCopyUniformCudaHip<TDim, TViewDst, TViewSrc, TExtent>(
                    viewDst,
                    viewSrc,
                    extent,
                    ALPAKA_API_PREFIX(MemcpyHostToDevice),
                    iDevice,
                    iDevice);
            }
        };
        //#############################################################################
        //! The CUDA/HIP to CUDA/HIP memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevUniformCudaHipRt, DevUniformCudaHipRt>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::TaskCopyUniformCudaHip<TDim, TViewDst, TViewSrc, TExtent>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::detail::TaskCopyUniformCudaHip<TDim, TViewDst, TViewSrc, TExtent>(
                    viewDst,
                    viewSrc,
                    extent,
                    ALPAKA_API_PREFIX(MemcpyDeviceToDevice),
                    getDev(viewDst).m_iDevice,
                    getDev(viewSrc).m_iDevice);
            }
        };

        //#############################################################################
        //! The CUDA/HIP non-blocking device queue 1D copy enqueue trait specialization.
        template<typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking,
            alpaka::detail::TaskCopyUniformCudaHip<DimInt<1u>, TViewDst, TViewSrc, TExtent>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking& queue,
                alpaka::detail::TaskCopyUniformCudaHip<DimInt<1u>, TViewDst, TViewSrc, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };
        //#############################################################################
        //! The CUDA/HIP blocking device queue 1D copy enqueue trait specialization.
        template<typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking,
            alpaka::detail::TaskCopyUniformCudaHip<DimInt<1u>, TViewDst, TViewSrc, TExtent>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking& queue,
                alpaka::detail::TaskCopyUniformCudaHip<DimInt<1u>, TViewDst, TViewSrc, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(StreamSynchronize)(queue.m_spQueueImpl->m_UniformCudaHipQueue));
            }
        };
        //#############################################################################
        //! The CUDA/HIP non-blocking device queue 2D copy enqueue trait specialization.
        template<typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking,
            alpaka::detail::TaskCopyUniformCudaHip<DimInt<2u>, TViewDst, TViewSrc, TExtent>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking& queue,
                alpaka::detail::TaskCopyUniformCudaHip<DimInt<2u>, TViewDst, TViewSrc, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };
        //#############################################################################
        //! The CUDA/HIP blocking device queue 2D copy enqueue trait specialization.
        template<typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking,
            alpaka::detail::TaskCopyUniformCudaHip<DimInt<2u>, TViewDst, TViewSrc, TExtent>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking& queue,
                alpaka::detail::TaskCopyUniformCudaHip<DimInt<2u>, TViewDst, TViewSrc, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(StreamSynchronize)(queue.m_spQueueImpl->m_UniformCudaHipQueue));
            }
        };
        //#############################################################################
        //! The CUDA/HIP non-blocking device queue 3D copy enqueue trait specialization.
        template<typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtNonBlocking,
            alpaka::detail::TaskCopyUniformCudaHip<DimInt<3u>, TViewDst, TViewSrc, TExtent>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtNonBlocking& queue,
                alpaka::detail::TaskCopyUniformCudaHip<DimInt<3u>, TViewDst, TViewSrc, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);
            }
        };
        //#############################################################################
        //! The CUDA/HIP blocking device queue 3D copy enqueue trait specialization.
        template<typename TExtent, typename TViewSrc, typename TViewDst>
        struct Enqueue<
            QueueUniformCudaHipRtBlocking,
            alpaka::detail::TaskCopyUniformCudaHip<DimInt<3u>, TViewDst, TViewSrc, TExtent>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(
                QueueUniformCudaHipRtBlocking& queue,
                alpaka::detail::TaskCopyUniformCudaHip<DimInt<3u>, TViewDst, TViewSrc, TExtent> const& task) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                task.enqueue(queue);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ALPAKA_API_PREFIX(StreamSynchronize)(queue.m_spQueueImpl->m_UniformCudaHipQueue));
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
