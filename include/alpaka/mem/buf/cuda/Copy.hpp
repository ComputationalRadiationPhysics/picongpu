/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/stream/StreamCudaRtSync.hpp>
#include <alpaka/stream/StreamCudaRtAsync.hpp>

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevCudaRt.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/stream/StreamCudaRtAsync.hpp>
#include <alpaka/stream/StreamCudaRtSync.hpp>

#include <alpaka/core/Cuda.hpp>

#include <cassert>

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
                    //! The CUDA memory copy trait.
                    template<
                        typename TDim,
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopy;

                    //#############################################################################
                    //! The 1D CUDA memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopy<
                        dim::DimInt<1>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Size = size::Size<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopy(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            cudaMemcpyKind const & cudaMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_cudaMemCpyKind(cudaMemCpyKind),
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extent)),
                                m_dstWidth(static_cast<Size>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Size>(extent::getWidth(viewSrc))),
#endif
                                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Size>(sizeof(elem::Elem<TViewDst>))),
                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentWidth <= m_srcWidth);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
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
                        cudaMemcpyKind m_cudaMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_extentWidth;
                        Size m_dstWidth;
                        Size m_srcWidth;
#endif
                        Size m_extentWidthBytes;
                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 2D CUDA memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopy<
                        dim::DimInt<2>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Size = size::Size<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopy(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            cudaMemcpyKind const & cudaMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_cudaMemCpyKind(cudaMemCpyKind),
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extent)),
#endif
                                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Size>(sizeof(elem::Elem<TViewDst>))),
                                m_dstWidth(static_cast<Size>(extent::getWidth(viewDst))),      // required for 3D peer copy
                                m_srcWidth(static_cast<Size>(extent::getWidth(viewSrc))),      // required for 3D peer copy

                                m_extentHeight(extent::getHeight(extent)),
                                m_dstHeight(static_cast<Size>(extent::getHeight(viewDst))),    // required for 3D peer copy
                                m_srcHeight(static_cast<Size>(extent::getHeight(viewSrc))),    // required for 3D peer copy

                                m_dstpitchBytesX(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - 1u>(viewDst))),
                                m_srcpitchBytesX(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - 1u>(viewSrc))),
                                m_dstPitchBytesY(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - (2u % dim::Dim<TViewDst>::value)>(viewDst))),
                                m_srcPitchBytesY(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - (2u % dim::Dim<TViewDst>::value)>(viewSrc))),

                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentHeight <= m_dstHeight);
                            assert(m_extentWidth <= m_srcWidth);
                            assert(m_extentHeight <= m_srcHeight);
                            assert(m_extentWidthBytes <= m_dstpitchBytesX);
                            assert(m_extentWidthBytes <= m_srcpitchBytesX);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
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
                        cudaMemcpyKind m_cudaMemCpyKind;
                        int m_iDstDevice;
                        int m_iSrcDevice;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_extentWidth;
#endif
                        Size m_extentWidthBytes;
                        Size m_dstWidth;          // required for 3D peer copy
                        Size m_srcWidth;          // required for 3D peer copy

                        Size m_extentHeight;
                        Size m_dstHeight;         // required for 3D peer copy
                        Size m_srcHeight;         // required for 3D peer copy

                        Size m_dstpitchBytesX;
                        Size m_srcpitchBytesX;
                        Size m_dstPitchBytesY;
                        Size m_srcPitchBytesY;


                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 3D CUDA memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopy<
                        dim::DimInt<3>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Size of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Size = size::Size<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopy(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            cudaMemcpyKind const & cudaMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_cudaMemCpyKind(cudaMemCpyKind),

                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),

                                m_extentWidth(extent::getWidth(extent)),
                                m_extentWidthBytes(m_extentWidth * static_cast<Size>(sizeof(elem::Elem<TViewDst>))),
                                m_dstWidth(static_cast<Size>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Size>(extent::getWidth(viewSrc))),

                                m_extentHeight(extent::getHeight(extent)),
                                m_dstHeight(static_cast<Size>(extent::getHeight(viewDst))),
                                m_srcHeight(static_cast<Size>(extent::getHeight(viewSrc))),

                                m_extentDepth(extent::getDepth(extent)),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_dstDepth(static_cast<Size>(extent::getDepth(viewDst))),
                                m_srcDepth(static_cast<Size>(extent::getDepth(viewSrc))),
#endif
                                m_dstpitchBytesX(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - 1u>(viewDst))),
                                m_srcpitchBytesX(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - 1u>(viewSrc))),
                                m_dstPitchBytesY(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - (2u % dim::Dim<TViewDst>::value)>(viewDst))),
                                m_srcPitchBytesY(static_cast<Size>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - (2u % dim::Dim<TViewDst>::value)>(viewSrc))),


                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            assert(m_extentWidth <= m_dstWidth);
                            assert(m_extentHeight <= m_dstHeight);
                            assert(m_extentDepth <= m_dstDepth);
                            assert(m_extentWidth <= m_srcWidth);
                            assert(m_extentHeight <= m_srcHeight);
                            assert(m_extentDepth <= m_srcDepth);
                            assert(m_extentWidthBytes <= m_dstpitchBytesX);
                            assert(m_extentWidthBytes <= m_srcpitchBytesX);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
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
                        cudaMemcpyKind m_cudaMemCpyKind;

                        int m_iDstDevice;
                        int m_iSrcDevice;

                        Size m_extentWidth;
                        Size m_extentWidthBytes;
                        Size m_dstWidth;
                        Size m_srcWidth;

                        Size m_extentHeight;
                        Size m_dstHeight;
                        Size m_srcHeight;

                        Size m_extentDepth;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Size m_dstDepth;
                        Size m_srcDepth;
#endif
                        Size m_dstpitchBytesX;
                        Size m_srcpitchBytesX;
                        Size m_dstPitchBytesY;
                        Size m_srcPitchBytesY;

                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                }
            }

            //-----------------------------------------------------------------------------
            // Trait specializations for TaskCopy.
            namespace traits
            {
                //#############################################################################
                //! The CUDA to CPU memory copy trait specialization.
                template<
                    typename TDim>
                struct TaskCopy<
                    TDim,
                    dev::DevCpu,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto taskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::cuda::detail::TaskCopy<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(viewSrc).m_iDevice);

                        return
                            mem::view::cuda::detail::TaskCopy<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    cudaMemcpyDeviceToHost,
                                    iDevice,
                                    iDevice);
                    }
                };
                //#############################################################################
                //! The CPU to CUDA memory copy trait specialization.
                template<
                    typename TDim>
                struct TaskCopy<
                    TDim,
                    dev::DevCudaRt,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto taskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::cuda::detail::TaskCopy<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(viewDst).m_iDevice);

                        return
                            mem::view::cuda::detail::TaskCopy<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    cudaMemcpyHostToDevice,
                                    iDevice,
                                    iDevice);
                    }
                };
                //#############################################################################
                //! The CUDA to CUDA memory copy trait specialization.
                template<
                    typename TDim>
                struct TaskCopy<
                    TDim,
                    dev::DevCudaRt,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto taskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::cuda::detail::TaskCopy<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::cuda::detail::TaskCopy<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    cudaMemcpyDeviceToDevice,
                                    dev::getDev(viewDst).m_iDevice,
                                    dev::getDev(viewSrc).m_iDevice);
                    }
                };
            }
            namespace cuda
            {
                namespace detail
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST auto buildCudaMemcpy3DParms(
                        mem::view::cuda::detail::TaskCopy<dim::DimInt<3>, TViewDst, TViewSrc, TExtent> const & task)
                    -> cudaMemcpy3DParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & dstWidth(task.m_dstWidth);
                        auto const & srcWidth(task.m_srcWidth);

                        auto const & extentHeight(task.m_extentHeight);
                        //auto const & dstHeight(task.m_dstHeight);
                        //auto const & srcHeight(task.m_srcHeight);

                        auto const & extentDepth(task.m_extentDepth);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);
                        auto const & dstPitchBytesY(task.m_dstPitchBytesY);
                        auto const & srcPitchBytesY(task.m_srcPitchBytesY);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DParms cudaMemCpy3DParms;
                        cudaMemCpy3DParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                        cudaMemCpy3DParms.srcPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DParms.srcPtr =
                            make_cudaPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(srcWidth),
                                static_cast<std::size_t>(srcPitchBytesY/srcPitchBytesX));
                        cudaMemCpy3DParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                        cudaMemCpy3DParms.dstPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DParms.dstPtr =
                            make_cudaPitchedPtr(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                static_cast<std::size_t>(dstWidth),
                                static_cast<std::size_t>(dstPitchBytesY/dstPitchBytesX));
                        cudaMemCpy3DParms.extent =
                            make_cudaExtent(
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                static_cast<std::size_t>(extentDepth));
                        cudaMemCpy3DParms.kind = task.m_cudaMemCpyKind;

                        return cudaMemCpy3DParms;
                    }
                    //-----------------------------------------------------------------------------
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    ALPAKA_FN_HOST auto buildCudaMemcpy3DPeerParms(
                        mem::view::cuda::detail::TaskCopy<dim::DimInt<2>, TViewDst, TViewSrc, TExtent> const & task)
                    -> cudaMemcpy3DPeerParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const & iDstDev(task.m_iDstDevice);
                        auto const & iSrcDev(task.m_iSrcDevice);

                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & dstWidth(task.m_dstWidth);
                        auto const & srcWidth(task.m_srcWidth);

                        auto const & extentHeight(task.m_extentHeight);
                        //auto const & dstHeight(task.m_dstHeight);
                        //auto const & srcHeight(task.m_srcHeight);

                        auto const extentDepth(1u);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);
                        auto const & dstPitchBytesY(task.m_dstPitchBytesY);
                        auto const & srcPitchBytesY(task.m_srcPitchBytesY);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms;
                        cudaMemCpy3DPeerParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                        cudaMemCpy3DPeerParms.dstDevice = iDstDev;
                        cudaMemCpy3DPeerParms.dstPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.dstPtr =
                            make_cudaPitchedPtr(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                static_cast<std::size_t>(dstWidth),
                                static_cast<std::size_t>(dstPitchBytesY/dstPitchBytesX));
                        cudaMemCpy3DPeerParms.extent =
                            make_cudaExtent(
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                static_cast<std::size_t>(extentDepth));
                        cudaMemCpy3DPeerParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                        cudaMemCpy3DPeerParms.srcDevice = iSrcDev;
                        cudaMemCpy3DPeerParms.srcPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.srcPtr =
                            make_cudaPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(srcWidth),
                                static_cast<std::size_t>(srcPitchBytesY/srcPitchBytesX));

                        return cudaMemCpy3DPeerParms;
                    }
                    //-----------------------------------------------------------------------------
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    ALPAKA_FN_HOST auto buildCudaMemcpy3DPeerParms(
                        mem::view::cuda::detail::TaskCopy<dim::DimInt<3>, TViewDst, TViewSrc, TExtent> const & task)
                    -> cudaMemcpy3DPeerParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const & iDstDev(task.m_iDstDevice);
                        auto const & iSrcDev(task.m_iSrcDevice);

                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & dstWidth(task.m_dstWidth);
                        auto const & srcWidth(task.m_srcWidth);

                        auto const & extentHeight(task.m_extentHeight);
                        //auto const & dstHeight(task.m_dstHeight);
                        //auto const & srcHeight(task.m_srcHeight);

                        auto const & extentDepth(task.m_extentDepth);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);
                        auto const & dstPitchBytesY(task.m_dstPitchBytesY);
                        auto const & srcPitchBytesY(task.m_srcPitchBytesY);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms;
                        cudaMemCpy3DPeerParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                        cudaMemCpy3DPeerParms.dstDevice = iDstDev;
                        cudaMemCpy3DPeerParms.dstPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.dstPtr =
                            make_cudaPitchedPtr(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                static_cast<std::size_t>(dstWidth),
                                static_cast<std::size_t>(dstPitchBytesY/dstPitchBytesX));
                        cudaMemCpy3DPeerParms.extent =
                            make_cudaExtent(
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                static_cast<std::size_t>(extentDepth));
                        cudaMemCpy3DPeerParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                        cudaMemCpy3DPeerParms.srcDevice = iSrcDev;
                        cudaMemCpy3DPeerParms.srcPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.srcPtr =
                            make_cudaPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(srcWidth),
                                static_cast<std::size_t>(srcPitchBytesY/srcPitchBytesX));

                        return cudaMemCpy3DPeerParms;
                    }
                }
            }
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA async device stream 1D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    auto const & extentWidthBytes(task.m_extentWidthBytes);

                    auto const & dstNativePtr(task.m_dstMemNative);
                    auto const & srcNativePtr(task.m_srcMemNative);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyAsync(
                                dstNativePtr,
                                srcNativePtr,
                                static_cast<std::size_t>(extentWidthBytes),
                                cudaMemCpyKind,
                                stream.m_spStreamImpl->m_CudaStream));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeerAsync(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                static_cast<std::size_t>(extentWidthBytes),
                                stream.m_spStreamImpl->m_CudaStream));
                    }
                }
            };
            //#############################################################################
            //! The CUDA sync device stream 1D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                stream::StreamCudaRtSync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync &,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    auto const & extentWidthBytes(task.m_extentWidthBytes);

                    auto const & dstNativePtr(task.m_dstMemNative);
                    auto const & srcNativePtr(task.m_srcMemNative);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy(
                                dstNativePtr,
                                srcNativePtr,
                                static_cast<std::size_t>(extentWidthBytes),
                                cudaMemCpyKind));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeer(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                static_cast<std::size_t>(extentWidthBytes)));
                    }
                }
            };
            //#############################################################################
            //! The CUDA async device stream 2D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & extentHeight(task.m_extentHeight);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2DAsync(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                srcNativePtr,
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                cudaMemCpyKind,
                                stream.m_spStreamImpl->m_CudaStream));
                    }
                    else
                    {
                        // There is no cudaMemcpy2DPeerAsync, therefore we use cudaMemcpy3DPeerAsync.
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeerAsync(
                                &cudaMemCpy3DPeerParms,
                                stream.m_spStreamImpl->m_CudaStream));
                    }
                }
            };
            //#############################################################################
            //! The CUDA sync device stream 2D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                stream::StreamCudaRtSync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync &,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & extentHeight(task.m_extentHeight);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        auto const & cudaMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2D(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                srcNativePtr,
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                cudaMemCpyKind));
                    }
                    else
                    {
                        // There is no cudaMemcpy2DPeerAsync, therefore we use cudaMemcpy3DPeerAsync.
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeer(
                                &cudaMemCpy3DPeerParms));
                    }
                }
            };
            //#############################################################################
            //! The CUDA async device stream 3D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                stream::StreamCudaRtAsync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtAsync & stream,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DParms const cudaMemCpy3DParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DParms(
                                task));
                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DAsync(
                                &cudaMemCpy3DParms,
                                stream.m_spStreamImpl->m_CudaStream));
                    }
                    else
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeerAsync(
                                &cudaMemCpy3DPeerParms,
                                stream.m_spStreamImpl->m_CudaStream));
                    }
                }
            };
            //#############################################################################
            //! The CUDA sync device stream 3D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                stream::StreamCudaRtSync,
                mem::view::cuda::detail::TaskCopy<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    stream::StreamCudaRtSync &,
                    mem::view::cuda::detail::TaskCopy<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DParms const cudaMemCpy3DParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DParms(
                                task));
                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3D(
                                &cudaMemCpy3DParms));
                    }
                    else
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::cuda::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeer(
                                &cudaMemCpy3DPeerParms));
                    }
                }
            };
        }
    }
}

#endif
