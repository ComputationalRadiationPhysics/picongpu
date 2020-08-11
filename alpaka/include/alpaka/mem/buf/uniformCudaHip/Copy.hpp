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

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>
#include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevUniformCudaHipRt.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>
#include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>

#include <alpaka/core/Assert.hpp>
// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    #include <alpaka/core/Cuda.hpp>
#else
    #include <alpaka/core/Hip.hpp>
#endif

#include <set>
#include <tuple>
#include <cstdint>
#include <type_traits>

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace uniform_cuda_hip
            {
                namespace detail
                {
                    using vec3D = alpaka::vec::Vec<alpaka::dim::DimInt<3u>, size_t>;
                    using vec2D = alpaka::vec::Vec<alpaka::dim::DimInt<2u>, size_t>;

                    ///! copy 3D memory
                    ///
                    /// It is required to start  `height * depth` HIP/CUDA blocks.
                    /// The kernel loops over the memory rows.
                    template<typename T>
                    __global__ void hipMemcpy3DEmulatedKernelD2D(
                        char * dstPtr, vec2D const dstPitch,
                        char const * srcPtr, vec2D const srcPitch, vec3D const extent
                    )
                    {
                        constexpr size_t X = 2;
                        constexpr size_t Y = 1;
                        constexpr size_t Z = 0;

                        // blockDim.[y,z] is always 1 and is not needed for index calculations
                        // gridDim and blockIdx is already in alpaka index order [z,y,x]
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        alpaka::vec::Vec<alpaka::dim::DimInt<3u>, uint32_t> const tid(
                            blockIdx.x,
                            blockIdx.y,
                            blockIdx.z * blockDim.x + threadIdx.x);

                        size_t const bytePerBlock = sizeof(T) * blockDim.x;
                        size_t const bytePerRow = gridDim.z * bytePerBlock;
#else
                        alpaka::vec::Vec<alpaka::dim::DimInt<3u>, uint32_t> const tid(
                            hipBlockIdx_x,
                            hipBlockIdx_y,
                            hipBlockIdx_z * hipBlockDim_x + hipThreadIdx_x);

                        size_t const bytePerBlock = sizeof(T) * hipBlockDim_x;
                        size_t const bytePerRow = hipGridDim_z * bytePerBlock;
#endif

                        size_t const peelLoopSteps = extent[X] / bytePerRow;
                        bool const needRemainder = (extent[X] % bytePerRow) != 0;

                        dstPtr += tid[Z] * dstPitch[Z] + tid[Y] * dstPitch[Y];
                        srcPtr += tid[Z] * srcPitch[Z] + tid[Y] * srcPitch[Y];

                        for(size_t idx = 0; idx < peelLoopSteps; ++idx)
                        {
                            size_t const byteOffsetX = idx * bytePerRow + tid[X] * sizeof(T);
                            auto dst = reinterpret_cast<T *>(dstPtr + byteOffsetX);
                            auto src = reinterpret_cast<T const *>(srcPtr + byteOffsetX);
                                *dst = *src;
                        }
                        if(needRemainder)
                        {
                            size_t const byteOffsetX = peelLoopSteps * bytePerRow + tid[X] * sizeof(T);
                            if(byteOffsetX < extent[X])
                            {
                                auto dst = reinterpret_cast<T *>(dstPtr + byteOffsetX);
                                auto src = reinterpret_cast<T const *>(srcPtr + byteOffsetX);
                                *dst = *src;
                            }
                        }
                    }

                    inline size_t divUp(size_t const & x, size_t const & y)
                    {
                        return (x + y - 1u) / y;
                    }

                    inline auto memcpy3DEmulatedD2DAsync(ALPAKA_API_PREFIX(Memcpy3DParms) const * const p, ALPAKA_API_PREFIX(Stream_t) stream)
                    {
                        using dim3Value_t = std::remove_reference_t<decltype(std::declval<dim3>().x)>;
                        // extent[2] is in byte
                        vec3D const extent(p->extent.depth, p->extent.height, p->extent.width);
                        // pitch in bytes
                        vec2D const dstPitch(p->dstPtr.pitch * p->dstPtr.ysize,p->dstPtr.pitch);
                        vec2D const srcPitch(p->srcPtr.pitch * p->srcPtr.ysize,p->srcPtr.pitch);
                        // offset[2] is in byte
                        vec3D const dstOffset(p->dstPos.z,p->dstPos.y,p->dstPos.x);
                        vec3D const srcOffset(p->srcPos.z,p->srcPos.y,p->srcPos.x);

                        char const * srcPtr =
                             reinterpret_cast<char const *>(p->srcPtr.ptr) + srcOffset[0] * srcPitch[0] + srcOffset[1] * srcPitch[1] + srcOffset[2];
                        char * dstPtr =
                             reinterpret_cast<char *>(p->dstPtr.ptr) + dstOffset[0] * dstPitch[0] + dstOffset[1] * dstPitch[1] + dstOffset[2];

                        bool const use4Byte = (reinterpret_cast<uintptr_t>(srcPtr) % 4u) == 0u && (reinterpret_cast<uintptr_t>(dstPtr) % 4u) == 0u && (extent[2] % 4u) == 0u;

                        if(use4Byte)
                        {
                            dim3 block(static_cast<dim3Value_t>(std::min(divUp(extent[2], 4u), size_t(256u))));
                            // use alpaka index order [z,y,x] because x is by default 1
                            dim3 grid(static_cast<dim3Value_t>(extent[0]),  static_cast<dim3Value_t>(extent[1]), 1u);
                            // for less than 100 blocks increase the number of blocks used to copy a row to
                            // increase the utilization of the device
                            if(grid.x * grid.y < 100)
                            {
                                grid.z = static_cast<dim3Value_t>(divUp(divUp(extent[2], 4u), block.x));
                            }

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            hipMemcpy3DEmulatedKernelD2D<int><<<
                                grid, block, 0, stream>>>(
                                    dstPtr, dstPitch, srcPtr, srcPitch, extent);
#else
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(hipMemcpy3DEmulatedKernelD2D<int>),
                                grid, block, 0, stream,
                                dstPtr, dstPitch, srcPtr, srcPitch, extent);
#endif
                        }
                        else
                        {
                            dim3 block(static_cast<dim3Value_t>(std::min(extent[2], size_t(256u))));
                            // use alpaka index order [z,y,x] because x is by default 1
                            dim3 grid(static_cast<dim3Value_t>(extent[0]), static_cast<dim3Value_t>(extent[1]), 1u);
                            // for less than 100 blocks increase the number of blocks used to copy a row to
                            // increase the utilization of the device
                            if(grid.x * grid.y < 100)
                            {
                                grid.z = static_cast<dim3Value_t>(divUp(extent[2], block.x));
                            }
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            hipMemcpy3DEmulatedKernelD2D<char><<<
                                grid, block, 0, stream>>>(
                                    dstPtr, dstPitch, srcPtr, srcPitch, extent);
#else
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(hipMemcpy3DEmulatedKernelD2D<char>),
                                grid, block, 0, stream,
                                dstPtr, dstPitch, srcPtr, srcPitch, extent);
#endif
                        }
                        return ALPAKA_API_PREFIX(GetLastError)();
                    };

                    //-----------------------------------------------------------------------------
                    //! Not being able to enable peer access does not prevent such device to device memory copies.
                    //! However, those copies may be slower because the memory is copied via the CPU.
                    inline auto enablePeerAccessIfPossible(
                        const int & devSrc,
                        const int & devDst)
                    -> void
                    {
                        ALPAKA_ASSERT(devSrc != devDst);

#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
                        static std::set<std::pair<int, int>> alreadyCheckedPeerAccessDevices;
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                        auto const devicePair = std::make_pair(devSrc, devDst);

                        if(alreadyCheckedPeerAccessDevices.find(devicePair) == alreadyCheckedPeerAccessDevices.end())
                        {
                            alreadyCheckedPeerAccessDevices.insert(devicePair);

                            int canAccessPeer = 0;
                            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(DeviceCanAccessPeer)(&canAccessPeer, devSrc, devDst));

                            if(!canAccessPeer) {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            std::cout << __func__
                                << " Direct peer access between given GPUs is not possible!"
                                << " src=" << devSrc
                                << " dst=" << devDst
                                << std::endl;
#endif
                                return;
                            }
                            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(SetDevice)(devSrc));

                            // NOTE: "until access is explicitly disabled using cudaDeviceDisablePeerAccess() or either device is reset using cudaDeviceReset()."
                            // We do not remove a device from the enabled device pairs on cudaDeviceReset.
                            // Note that access granted by this call is unidirectional and that in order to access memory on the current device from peerDevice, a separate symmetric call to cudaDeviceEnablePeerAccess() is required.
                            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ALPAKA_API_PREFIX(DeviceEnablePeerAccess)(devDst, 0));
                        }
                    }

                    //#############################################################################
                    //! The CUDA/HIP memory copy trait.
                    template<
                        typename TDim,
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyUniformCudaHip;

                    //#############################################################################
                    //! The 1D CUDA/HIP memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyUniformCudaHip<
                        dim::DimInt<1>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        using MemcpyKind = ALPAKA_API_PREFIX(MemcpyKind);

                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, std::remove_const_t<elem::Elem<TViewSrc>>>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            MemcpyKind const & uniformMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_uniformMemCpyKind(uniformMemCpyKind),
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extent)),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),
#endif
                                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                            ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
#endif
                        }

                        //-----------------------------------------------------------------------------
                        template<
                            typename TQueue
                        >
                        auto enqueue(TQueue & queue) const
                        -> void
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDebug();
#endif
                            if(m_extentWidthBytes == 0)
                            {
                                return;
                            }

                            auto const & uniformCudaHipMemCpyKind(m_uniformMemCpyKind);

                            if(m_iDstDevice == m_iSrcDevice)
                            {
                                // Set the current device.
                                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                    ALPAKA_API_PREFIX(SetDevice)(
                                        m_iDstDevice));
                                // Initiate the memory copy.
                                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                    ALPAKA_API_PREFIX(MemcpyAsync)(
                                        m_dstMemNative,
                                        m_srcMemNative,
                                        static_cast<std::size_t>(m_extentWidthBytes),
                                        uniformCudaHipMemCpyKind,
                                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
                            }
                            else
                            {
                                alpaka::mem::view::uniform_cuda_hip::detail::enablePeerAccessIfPossible(m_iSrcDevice, m_iDstDevice);

                                // Initiate the memory copy.
                                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                    ALPAKA_API_PREFIX(MemcpyPeerAsync)(
                                        m_dstMemNative,
                                        m_iDstDevice,
                                        m_srcMemNative,
                                        m_iSrcDevice,
                                        static_cast<std::size_t>(m_extentWidthBytes),
                                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
                            }
                        }

                    private:
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << __func__
                                << " ddev: " << m_iDstDevice
                                << " ew: " << m_extentWidth
                                << " ewb: " << m_extentWidthBytes
                                << " dw: " << m_dstWidth
                                << " dptr: " << m_dstMemNative
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sptr: " << m_srcMemNative
                                << std::endl;
                        }
#endif

                        MemcpyKind m_uniformMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_extentWidth;
                        Idx m_dstWidth;
                        Idx m_srcWidth;
#endif
                        Idx m_extentWidthBytes;
                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 2D CUDA/HIP memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyUniformCudaHip<
                        dim::DimInt<2>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        using MemcpyKind = ALPAKA_API_PREFIX(MemcpyKind);

                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, std::remove_const_t<elem::Elem<TViewSrc>>>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            MemcpyKind const & uniformMemcpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_uniformMemCpyKind(uniformMemcpyKind),
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extent)),
#endif
                                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),      // required for 3D peer copy
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),      // required for 3D peer copy

                                m_extentHeight(extent::getHeight(extent)),
                                m_dstHeight(static_cast<Idx>(extent::getHeight(viewDst))),    // required for 3D peer copy
                                m_srcHeight(static_cast<Idx>(extent::getHeight(viewSrc))),    // required for 3D peer copy

                                m_dstpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - 1u>(viewDst))),
                                m_srcpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - 1u>(viewSrc))),
                                m_dstPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - (2u % dim::Dim<TViewDst>::value)>(viewDst))),
                                m_srcPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - (2u % dim::Dim<TViewDst>::value)>(viewSrc))),

                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                            ALPAKA_ASSERT(m_extentHeight <= m_dstHeight);
                            ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
                            ALPAKA_ASSERT(m_extentHeight <= m_srcHeight);
                            ALPAKA_ASSERT(m_extentWidthBytes <= m_dstpitchBytesX);
                            ALPAKA_ASSERT(m_extentWidthBytes <= m_srcpitchBytesX);
#endif
                        }

                        //-----------------------------------------------------------------------------
                        template<
                            typename TQueue
                        >
                        auto enqueue(TQueue & queue) const
                        -> void
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDebug();
#endif
                            // This is not only an optimization but also prevents a division by zero.
                            if(m_extentWidthBytes == 0 || m_extentHeight == 0)
                            {
                                return;
                            }

                            if(m_iDstDevice == m_iSrcDevice)
                            {
                                // Set the current device.
                                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                    ALPAKA_API_PREFIX(SetDevice)(
                                        m_iDstDevice));
                                // Initiate the memory copy.
                                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                    ALPAKA_API_PREFIX(Memcpy2DAsync)(
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
                                alpaka::mem::view::uniform_cuda_hip::detail::enablePeerAccessIfPossible(m_iSrcDevice, m_iDstDevice);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                                // There is no cudaMemcpy2DPeerAsync, therefore we use cudaMemcpy3DPeerAsync.
                                // Create the struct describing the copy.
                                ALPAKA_API_PREFIX(Memcpy3DPeerParms) const memCpy3DPeerParms(
                                    buildCudaMemcpy3DPeerParms());
                                // Initiate the memory copy.
                                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                    cudaMemcpy3DPeerAsync(
                                        &memCpy3DPeerParms,
                                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
#endif
                            }
                        }

                    private:
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto buildCudaMemcpy3DPeerParms() const
                        -> cudaMemcpy3DPeerParms
                        {
                            ALPAKA_DEBUG_FULL_LOG_SCOPE;

                            // Fill CUDA parameter structure.
                            cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms;
                            cudaMemCpy3DPeerParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                            cudaMemCpy3DPeerParms.dstDevice = m_iDstDevice;
                            cudaMemCpy3DPeerParms.dstPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                            cudaMemCpy3DPeerParms.dstPtr =
                                make_cudaPitchedPtr(
                                    m_dstMemNative,
                                    static_cast<std::size_t>(m_dstpitchBytesX),
                                    static_cast<std::size_t>(m_dstWidth),
                                    static_cast<std::size_t>(m_dstPitchBytesY / m_dstpitchBytesX));
                            cudaMemCpy3DPeerParms.extent =
                                make_cudaExtent(
                                    static_cast<std::size_t>(m_extentWidthBytes),
                                    static_cast<std::size_t>(m_extentHeight),
                                    static_cast<std::size_t>(1u));
                            cudaMemCpy3DPeerParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                            cudaMemCpy3DPeerParms.srcDevice = m_iSrcDevice;
                            cudaMemCpy3DPeerParms.srcPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                            cudaMemCpy3DPeerParms.srcPtr =
                                make_cudaPitchedPtr(
                                    const_cast<void *>(m_srcMemNative),
                                    static_cast<std::size_t>(m_srcpitchBytesX),
                                    static_cast<std::size_t>(m_srcWidth),
                                    static_cast<std::size_t>(m_srcPitchBytesY / m_srcpitchBytesX));

                            return cudaMemCpy3DPeerParms;
                        }
#endif
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << __func__
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ewb: " << m_extentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dptr: " << m_dstMemNative
                                << " dpitchb: " << m_dstpitchBytesX
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sptr: " << m_srcMemNative
                                << " spitchb: " << m_srcpitchBytesX
                                << std::endl;
                        }
#endif

                        MemcpyKind m_uniformMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_extentWidth;
#endif
                        Idx m_extentWidthBytes;
                        Idx m_dstWidth;          // required for 3D peer copy
                        Idx m_srcWidth;          // required for 3D peer copy

                        Idx m_extentHeight;
                        Idx m_dstHeight;         // required for 3D peer copy
                        Idx m_srcHeight;         // required for 3D peer copy

                        Idx m_dstpitchBytesX;
                        Idx m_srcpitchBytesX;
                        Idx m_dstPitchBytesY;
                        Idx m_srcPitchBytesY;


                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 3D CUDA/HIP memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyUniformCudaHip<
                        dim::DimInt<3>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        using MemcpyKind = ALPAKA_API_PREFIX(MemcpyKind);

                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, std::remove_const_t<elem::Elem<TViewSrc>>>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyUniformCudaHip(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            MemcpyKind const & uniformMemcpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_uniformMemCpyKind(uniformMemcpyKind),
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),

                                m_extentWidth(extent::getWidth(extent)),
                                m_extentWidthBytes(m_extentWidth * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),

                                m_extentHeight(extent::getHeight(extent)),
                                m_dstHeight(static_cast<Idx>(extent::getHeight(viewDst))),
                                m_srcHeight(static_cast<Idx>(extent::getHeight(viewSrc))),

                                m_extentDepth(extent::getDepth(extent)),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_dstDepth(static_cast<Idx>(extent::getDepth(viewDst))),
                                m_srcDepth(static_cast<Idx>(extent::getDepth(viewSrc))),
#endif
                                m_dstpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - 1u>(viewDst))),
                                m_srcpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - 1u>(viewSrc))),
                                m_dstPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - (2u % dim::Dim<TViewDst>::value)>(viewDst))),
                                m_srcPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - (2u % dim::Dim<TViewDst>::value)>(viewSrc))),


                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                            ALPAKA_ASSERT(m_extentHeight <= m_dstHeight);
                            ALPAKA_ASSERT(m_extentDepth <= m_dstDepth);
                            ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
                            ALPAKA_ASSERT(m_extentHeight <= m_srcHeight);
                            ALPAKA_ASSERT(m_extentDepth <= m_srcDepth);
                            ALPAKA_ASSERT(m_extentWidthBytes <= m_dstpitchBytesX);
                            ALPAKA_ASSERT(m_extentWidthBytes <= m_srcpitchBytesX);
#endif
                        }

                        //-----------------------------------------------------------------------------
                        template<
                            typename TQueue
                        >
                        auto enqueue(TQueue & queue) const
                        -> void
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDebug();
#endif
                            // This is not only an optimization but also prevents a division by zero.
                            if(m_extentWidthBytes == 0 || m_extentHeight == 0 || m_extentDepth == 0)
                            {
                                return;
                            }

                            if(m_iDstDevice == m_iSrcDevice)
                            {
                                // Create the struct describing the copy.
                                ALPAKA_API_PREFIX(Memcpy3DParms) const uniformCudaHipMemCpy3DParms(
                                    buildUniformCudaHipMemcpy3DParms());
                                // Set the current device.
                                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                    ALPAKA_API_PREFIX(SetDevice)(
                                        m_iDstDevice));
#if defined(ALPAKA_EMU_MEMCPY3D_ENABLED)
                                auto isDevice2DeviceCopy = m_uniformMemCpyKind == ALPAKA_API_PREFIX(MemcpyDeviceToDevice);
                                // contiguous memory can be copied by a single 1D memory copy
                                auto isContiguousMemory = uniformCudaHipMemCpy3DParms.extent.width == uniformCudaHipMemCpy3DParms.dstPtr.pitch &&
                                        uniformCudaHipMemCpy3DParms.extent.width == uniformCudaHipMemCpy3DParms.srcPtr.pitch;
                                if(isDevice2DeviceCopy && !isContiguousMemory)
                                {
                                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                        uniform_cuda_hip::detail::memcpy3DEmulatedD2DAsync(
                                            &uniformCudaHipMemCpy3DParms,
                                            queue.m_spQueueImpl->m_UniformCudaHipQueue));
                                }
                                else
#endif
                                {
                                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                        ALPAKA_API_PREFIX(Memcpy3DAsync)(
                                            &uniformCudaHipMemCpy3DParms,
                                            queue.m_spQueueImpl->m_UniformCudaHipQueue));
                                }
                            }
                            else
                            {
                                alpaka::mem::view::uniform_cuda_hip::detail::enablePeerAccessIfPossible(m_iSrcDevice, m_iDstDevice);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                                // Create the struct describing the copy.
                                cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                                    buildCudaMemcpy3DPeerParms());
                                // Initiate the memory copy.
                                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                                    cudaMemcpy3DPeerAsync(
                                        &cudaMemCpy3DPeerParms,
                                        queue.m_spQueueImpl->m_UniformCudaHipQueue));
#endif
                            }
                        }

                    private:
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto buildUniformCudaHipMemcpy3DParms() const
                        -> ALPAKA_API_PREFIX(Memcpy3DParms)
                        {
                            ALPAKA_DEBUG_FULL_LOG_SCOPE;

                            // Fill CUDA/HIP parameter structure.
                            ALPAKA_API_PREFIX(Memcpy3DParms) memCpy3DParms;
                            memCpy3DParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                            memCpy3DParms.srcPos = ALPAKA_PP_CONCAT(make_,ALPAKA_API_PREFIX(Pos))(0, 0, 0);  // Optional. Offset in bytes.
                            memCpy3DParms.srcPtr =
                                ALPAKA_PP_CONCAT(make_,ALPAKA_API_PREFIX(PitchedPtr))(
                                    const_cast<void *>(m_srcMemNative),
                                    static_cast<std::size_t>(m_srcpitchBytesX),
                                    static_cast<std::size_t>(m_srcWidth),
                                    static_cast<std::size_t>(m_srcPitchBytesY/m_srcpitchBytesX));
                            memCpy3DParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                            memCpy3DParms.dstPos = ALPAKA_PP_CONCAT(make_,ALPAKA_API_PREFIX(Pos))(0, 0, 0);  // Optional. Offset in bytes.
                            memCpy3DParms.dstPtr =
                                ALPAKA_PP_CONCAT(make_,ALPAKA_API_PREFIX(PitchedPtr))(
                                    m_dstMemNative,
                                    static_cast<std::size_t>(m_dstpitchBytesX),
                                    static_cast<std::size_t>(m_dstWidth),
                                    static_cast<std::size_t>(m_dstPitchBytesY / m_dstpitchBytesX));
                            memCpy3DParms.extent =
                                ALPAKA_PP_CONCAT(make_,ALPAKA_API_PREFIX(Extent))(
                                    static_cast<std::size_t>(m_extentWidthBytes),
                                    static_cast<std::size_t>(m_extentHeight),
                                    static_cast<std::size_t>(m_extentDepth));
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_PLATFORM_NVCC__)
                            memCpy3DParms.kind = hipMemcpyKindToCudaMemcpyKind(m_uniformMemCpyKind);
#else
                            memCpy3DParms.kind = m_uniformMemCpyKind;
#endif
                            return memCpy3DParms;
                        }

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto buildCudaMemcpy3DPeerParms() const
                        -> cudaMemcpy3DPeerParms
                        {
                            ALPAKA_DEBUG_FULL_LOG_SCOPE;

                            // Fill CUDA parameter structure.
                            cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms;
                            cudaMemCpy3DPeerParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                            cudaMemCpy3DPeerParms.dstDevice = m_iDstDevice;
                            cudaMemCpy3DPeerParms.dstPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                            cudaMemCpy3DPeerParms.dstPtr =
                                make_cudaPitchedPtr(
                                    m_dstMemNative,
                                    static_cast<std::size_t>(m_dstpitchBytesX),
                                    static_cast<std::size_t>(m_dstWidth),
                                    static_cast<std::size_t>(m_dstPitchBytesY/m_dstpitchBytesX));
                            cudaMemCpy3DPeerParms.extent =
                                make_cudaExtent(
                                    static_cast<std::size_t>(m_extentWidthBytes),
                                    static_cast<std::size_t>(m_extentHeight),
                                    static_cast<std::size_t>(m_extentDepth));
                            cudaMemCpy3DPeerParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                            cudaMemCpy3DPeerParms.srcDevice = m_iSrcDevice;
                            cudaMemCpy3DPeerParms.srcPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                            cudaMemCpy3DPeerParms.srcPtr =
                                make_cudaPitchedPtr(
                                    const_cast<void *>(m_srcMemNative),
                                    static_cast<std::size_t>(m_srcpitchBytesX),
                                    static_cast<std::size_t>(m_srcWidth),
                                    static_cast<std::size_t>(m_srcPitchBytesY / m_srcpitchBytesX));

                            return cudaMemCpy3DPeerParms;
                        }
#endif
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << __func__
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ed: " << m_extentDepth
                                << " ewb: " << m_extentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dd: " << m_dstDepth
                                << " dptr: " << m_dstMemNative
                                << " dpitchb: " << m_dstpitchBytesX
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sd: " << m_srcDepth
                                << " sptr: " << m_srcMemNative
                                << " spitchb: " << m_srcpitchBytesX
                                << std::endl;
                        }
#endif
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
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_dstDepth;
                        Idx m_srcDepth;
#endif
                        Idx m_dstpitchBytesX;
                        Idx m_srcpitchBytesX;
                        Idx m_dstPitchBytesY;
                        Idx m_srcPitchBytesY;

                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                }
            }

            //-----------------------------------------------------------------------------
            // Trait specializations for CreateTaskCopy.
            namespace traits
            {
                //#############################################################################
                //! The CUDA/HIP to CPU memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevCpu,
                    dev::DevUniformCudaHipRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(viewSrc).m_iDevice);

                        return
                            mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
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
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevUniformCudaHipRt,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(viewDst).m_iDevice);

                        return
                            mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
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
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevUniformCudaHipRt,
                    dev::DevUniformCudaHipRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    ALPAKA_API_PREFIX(MemcpyDeviceToDevice),
                                    dev::getDev(viewDst).m_iDevice,
                                    dev::getDev(viewSrc).m_iDevice);
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
            //! The CUDA/HIP non-blocking device queue 1D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformCudaHipRtNonBlocking,
                mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformCudaHipRtNonBlocking & queue,
                    mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    task.enqueue(queue);
                }
            };
            //#############################################################################
            //! The CUDA/HIP blocking device queue 1D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformCudaHipRtBlocking,
                mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformCudaHipRtBlocking & queue,
                    mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    task.enqueue(queue);

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(StreamSynchronize)(
                            queue.m_spQueueImpl->m_UniformCudaHipQueue));
                }
            };
            //#############################################################################
            //! The CUDA/HIP non-blocking device queue 2D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformCudaHipRtNonBlocking,
                mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformCudaHipRtNonBlocking & queue,
                    mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    task.enqueue(queue);
                }
            };
            //#############################################################################
            //! The CUDA/HIP blocking device queue 2D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformCudaHipRtBlocking,
                mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformCudaHipRtBlocking & queue,
                    mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    task.enqueue(queue);

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(StreamSynchronize)(
                            queue.m_spQueueImpl->m_UniformCudaHipQueue));
                }
            };
            //#############################################################################
            //! The CUDA/HIP non-blocking device queue 3D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformCudaHipRtNonBlocking,
                mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformCudaHipRtNonBlocking & queue,
                    mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    task.enqueue(queue);
                }
            };
            //#############################################################################
            //! The CUDA/HIP blocking device queue 3D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformCudaHipRtBlocking,
                mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformCudaHipRtBlocking & queue,
                    mem::view::uniform_cuda_hip::detail::TaskCopyUniformCudaHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    task.enqueue(queue);

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ALPAKA_API_PREFIX(StreamSynchronize)(
                            queue.m_spQueueImpl->m_UniformCudaHipQueue));
                }
            };
        }
    }
}

#endif
