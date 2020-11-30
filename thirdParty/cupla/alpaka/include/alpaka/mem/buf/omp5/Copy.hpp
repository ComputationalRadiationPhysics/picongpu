/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/core/Omp5.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/dev/DevOmp5.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/mem/view/Traits.hpp>
#    include <alpaka/queue/QueueOmp5Blocking.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <set>
#    include <tuple>
#    include <utility>

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The Omp5 memory copy trait.
        template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyOmp5
        {
            static_assert(!std::is_const<TViewDst>::value, "The destination view can not be const!");

            static_assert(
                Dim<TViewSrc>::value == TDim::value,
                "The source view is required to have dimensionality TDim!");
            static_assert(
                Dim<TViewDst>::value == Dim<TViewSrc>::value,
                "The source and the destination view are required to have the same dimensionality!");
            static_assert(
                Dim<TViewDst>::value == Dim<TExtent>::value,
                "The views and the extent are required to have the same dimensionality!");
            // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
            static_assert(
                std::is_same<Elem<TViewDst>, typename std::remove_const<Elem<TViewSrc>>::type>::value,
                "The source and the destination view are required to have the same element type!");

            using Idx = alpaka::Idx<TExtent>;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST TaskCopyOmp5(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                int const& iDstDevice,
                int const& iSrcDevice)
                : m_iDstDevice(iDstDevice)
                , m_iSrcDevice(iSrcDevice)
                , m_extent(castVec<size_t>(extent::getExtentVec(extent)))
                , m_dstPitchBytes(castVec<size_t>(getPitchBytesVec(viewDst)))
                , m_srcPitchBytes(castVec<size_t>(getPitchBytesVec(viewSrc)))
                , m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
            {
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                const auto dstExtent(castVec<size_t>(extent::getExtentVec(viewDst)));
                const auto srcExtent(castVec<size_t>(extent::getExtentVec(viewSrc)));
                for(auto i = static_cast<decltype(TDim::value)>(0u); i < TDim::value; ++i)
                {
                    ALPAKA_ASSERT(m_extent[i] <= dstExtent[i]);
                    ALPAKA_ASSERT(m_extent[i] <= srcExtent[i]);
                }
                std::cout << "TaskCopyOmp5<" << TDim::value << ",...>::ctor\tsrcExtent=" << srcExtent
                          << ", dstExtent=" << dstExtent << std::endl;
#    endif
            }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " ddev: " << m_iDstDevice << " ew: " << m_extent
                          << " dptr: " << m_dstMemNative << " sdev: " << m_iSrcDevice << " sptr: " << m_srcMemNative
                          << std::endl;
            }
#    endif
            int m_iDstDevice;
            int m_iSrcDevice;
            Vec<TDim, size_t> m_extent;
            Vec<TDim, size_t> m_dstPitchBytes;
            Vec<TDim, size_t> m_srcPitchBytes;
            void* m_dstMemNative;
            void const* m_srcMemNative;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            ALPAKA_FN_HOST auto operator()() const -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                constexpr auto lastDim = TDim::value - 1;

                if(m_extent.prod() > 0)
                {
                    // offsets == 0 by ptr shift (?)
                    auto dstOffset(Vec<TDim, size_t>::zeros());
                    auto srcOffset(Vec<TDim, size_t>::zeros());

                    auto dstExtentFull(Vec<TDim, size_t>::zeros());
                    auto srcExtentFull(Vec<TDim, size_t>::zeros());

                    const size_t elementSize
                        = (m_dstPitchBytes[0] % sizeof(Elem<TViewDst>) || m_srcPitchBytes[0] % sizeof(Elem<TViewDst>))
                        ? 1
                        : sizeof(Elem<TViewDst>);

                    dstExtentFull[lastDim] = m_dstPitchBytes[lastDim] / elementSize;
                    srcExtentFull[lastDim] = m_srcPitchBytes[lastDim] / elementSize;
                    for(int i = lastDim - 1; i >= 0; --i)
                    {
                        dstExtentFull[i] = m_dstPitchBytes[i] / m_dstPitchBytes[i + 1];
                        srcExtentFull[i] = m_srcPitchBytes[i] / m_srcPitchBytes[i + 1];
                    }

                    ALPAKA_OMP5_CHECK(omp_target_memcpy_rect(
                        m_dstMemNative,
                        const_cast<void*>(m_srcMemNative),
                        sizeof(Elem<TViewDst>),
                        TDim::value,
                        reinterpret_cast<size_t const*>(&m_extent),
                        reinterpret_cast<size_t const*>(&dstOffset),
                        reinterpret_cast<size_t const*>(&srcOffset),
                        reinterpret_cast<size_t const*>(&dstExtentFull),
                        reinterpret_cast<size_t const*>(&srcExtentFull),
                        m_iDstDevice,
                        m_iSrcDevice));
                }
            }
        };

        //#############################################################################
        //! The Omp5 memory copy trait.
        template<typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyOmp5<DimInt<1>, TViewDst, TViewSrc, TExtent>
        {
            static_assert(!std::is_const<TViewDst>::value, "The destination view can not be const!");

            static_assert(Dim<TViewSrc>::value == 1, "The source view is required to have dimensionality 1!");
            static_assert(Dim<TViewDst>::value == 1, "The source view is required to have dimensionality 1!");
            static_assert(Dim<TExtent>::value == 1, "The extent is required to have dimensionality 1!");
            // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
            static_assert(
                std::is_same<Elem<TViewDst>, typename std::remove_const<Elem<TViewSrc>>::type>::value,
                "The source and the destination view are required to have the same element type!");

            using Idx = alpaka::Idx<TExtent>;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST TaskCopyOmp5(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                int const& iDstDevice,
                int const& iSrcDevice)
                : m_iDstDevice(iDstDevice)
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

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            ALPAKA_FN_HOST auto operator()() const -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                if(m_extentWidthBytes == 0)
                {
                    return;
                }

                ALPAKA_OMP5_CHECK(omp_target_memcpy(
                    m_dstMemNative,
                    const_cast<void*>(m_srcMemNative),
                    static_cast<std::size_t>(m_extentWidthBytes),
                    0,
                    0,
                    m_iDstDevice,
                    m_iSrcDevice));
            }
        };
    } // namespace detail

    //-----------------------------------------------------------------------------
    // Trait specializations for CreateTaskMemcpy.
    namespace traits
    {
        namespace detail
        {
            //#############################################################################
            //! The Omp5 memory copy task creation trait detail.
            template<typename TDim, typename TDevDst, typename TDevSrc>
            struct CreateTaskCopyImpl
            {
                //-----------------------------------------------------------------------------
                template<typename TExtent, typename TViewSrc, typename TViewDst>
                ALPAKA_FN_HOST static auto createTaskMemcpy(
                    TViewDst& viewDst,
                    TViewSrc const& viewSrc,
                    TExtent const& extent,
                    int iDeviceDst = 0,
                    int iDeviceSrc = 0) -> alpaka::detail::TaskCopyOmp5<TDim, TViewDst, TViewSrc, TExtent>
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    return alpaka::detail::TaskCopyOmp5<TDim, TViewDst, TViewSrc, TExtent>(
                        viewDst,
                        viewSrc,
                        extent,
                        iDeviceDst,
                        iDeviceSrc);
                }
            };
        } // namespace detail

        //#############################################################################
        //! The CPU to Omp5 memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevOmp5, DevCpu>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::TaskCopyOmp5<TDim, TViewDst, TViewSrc, TExtent>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::detail::TaskCopyOmp5<TDim, TViewDst, TViewSrc, TExtent>(
                    viewDst,
                    viewSrc,
                    extent,
                    getDev(viewDst).m_spDevOmp5Impl->iDevice(),
                    omp_get_initial_device());
            }
        };

        //#############################################################################
        //! The Omp5 to CPU memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevCpu, DevOmp5>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::TaskCopyOmp5<TDim, TViewDst, TViewSrc, TExtent>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::detail::TaskCopyOmp5<TDim, TViewDst, TViewSrc, TExtent>(
                    viewDst,
                    viewSrc,
                    extent,
                    omp_get_initial_device(),
                    getDev(viewSrc).m_spDevOmp5Impl->iDevice());
            }
        };

        //#############################################################################
        //! The Omp5 to Omp5 memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevOmp5, DevOmp5>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::TaskCopyOmp5<TDim, TViewDst, TViewSrc, TExtent>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::detail::TaskCopyOmp5<TDim, TViewDst, TViewSrc, TExtent>(
                    viewDst,
                    viewSrc,
                    extent,
                    getDev(viewDst).m_spDevOmp5Impl->iDevice(),
                    getDev(viewSrc).m_spDevOmp5Impl->iDevice());
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
