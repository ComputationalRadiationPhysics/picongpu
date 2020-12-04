/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, Rene Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Assert.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/meta/Integral.hpp>
#include <alpaka/meta/NdLoop.hpp>

#include <cstring>

namespace alpaka
{
    class DevCpu;
}

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The CPU device memory copy task base.
        //!
        //! Copies from CPU memory into CPU memory.
        template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyCpuBase
        {
            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TViewDst>;
            using SrcSize = Idx<TViewSrc>;
            using Elem = alpaka::Elem<TViewSrc>;

            static_assert(!std::is_const<TViewDst>::value, "The destination view can not be const!");

            static_assert(
                Dim<TViewDst>::value == Dim<TViewSrc>::value,
                "The source and the destination view are required to have the same dimensionality!");
            static_assert(
                Dim<TViewDst>::value == Dim<TExtent>::value,
                "The views and the extent are required to have the same dimensionality!");
            static_assert(
                Dim<TViewDst>::value == TDim::value,
                "The destination view and the input TDim are required to have the same dimensionality!");

            static_assert(
                meta::IsIntegralSuperset<DstSize, ExtentSize>::value,
                "The destination view and the extent are required to have compatible idx type!");
            static_assert(
                meta::IsIntegralSuperset<SrcSize, ExtentSize>::value,
                "The source view and the extent are required to have compatible idx type!");

            static_assert(
                std::is_same<alpaka::Elem<TViewDst>, std::remove_const_t<alpaka::Elem<TViewSrc>>>::value,
                "The source and the destination view are required to have the same element type!");

            //-----------------------------------------------------------------------------
            TaskCopyCpuBase(TViewDst& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
                : m_extent(extent::getExtentVec(extent))
                , m_extentWidthBytes(m_extent[TDim::value - 1u] * static_cast<ExtentSize>(sizeof(Elem)))
                ,
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                m_dstExtent(extent::getExtentVec(viewDst))
                , m_srcExtent(extent::getExtentVec(viewSrc))
                ,
#endif
                m_dstPitchBytes(getPitchBytesVec(viewDst))
                , m_srcPitchBytes(getPitchBytesVec(viewSrc))
                ,

                m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<std::uint8_t const*>(getPtrNative(viewSrc)))
            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).foldrAll(std::logical_or<bool>()));
                ALPAKA_ASSERT((castVec<SrcSize>(m_extent) <= m_srcExtent).foldrAll(std::logical_or<bool>()));
                ALPAKA_ASSERT(static_cast<DstSize>(m_extentWidthBytes) <= m_dstPitchBytes[TDim::value - 1u]);
                ALPAKA_ASSERT(static_cast<SrcSize>(m_extentWidthBytes) <= m_srcPitchBytes[TDim::value - 1u]);
            }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " e: " << m_extent << " ewb: " << this->m_extentWidthBytes
                          << " de: " << m_dstExtent << " dptr: " << reinterpret_cast<void*>(m_dstMemNative)
                          << " dpitchb: " << m_dstPitchBytes << " se: " << m_srcExtent
                          << " sptr: " << reinterpret_cast<void const*>(m_srcMemNative)
                          << " spitchb: " << m_srcPitchBytes << std::endl;
            }
#endif

            Vec<TDim, ExtentSize> const m_extent;
            ExtentSize const m_extentWidthBytes;
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            Vec<TDim, DstSize> const m_dstExtent;
            Vec<TDim, SrcSize> const m_srcExtent;
#endif
            Vec<TDim, DstSize> const m_dstPitchBytes;
            Vec<TDim, SrcSize> const m_srcPitchBytes;

            std::uint8_t* const m_dstMemNative;
            std::uint8_t const* const m_srcMemNative;
        };


        //#############################################################################
        //! The CPU device ND memory copy task.
        template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyCpu : public TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>
        {
            using DimMin1 = DimInt<TDim::value - 1u>;
            using typename TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>::ExtentSize;
            using typename TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>::DstSize;
            using typename TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>::SrcSize;

            //-----------------------------------------------------------------------------
            using TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>::TaskCopyCpuBase;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator()() const -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#endif
                // [z, y, x] -> [z, y] because all elements with the innermost x dimension are handled within one
                // iteration.
                Vec<DimMin1, ExtentSize> const extentWithoutInnermost(subVecBegin<DimMin1>(this->m_extent));
                // [z, y, x] -> [y, x] because the z pitch (the full size of the buffer) is not required.
                Vec<DimMin1, DstSize> const dstPitchBytesWithoutOutmost(subVecEnd<DimMin1>(this->m_dstPitchBytes));
                Vec<DimMin1, SrcSize> const srcPitchBytesWithoutOutmost(subVecEnd<DimMin1>(this->m_srcPitchBytes));

                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    meta::ndLoopIncIdx(extentWithoutInnermost, [&](Vec<DimMin1, ExtentSize> const& idx) {
                        std::memcpy(
                            reinterpret_cast<void*>(
                                this->m_dstMemNative
                                + (castVec<DstSize>(idx) * dstPitchBytesWithoutOutmost)
                                      .foldrAll(std::plus<DstSize>())),
                            reinterpret_cast<void const*>(
                                this->m_srcMemNative
                                + (castVec<SrcSize>(idx) * srcPitchBytesWithoutOutmost)
                                      .foldrAll(std::plus<SrcSize>())),
                            static_cast<std::size_t>(this->m_extentWidthBytes));
                    });
                }
            }
        };

        //#############################################################################
        //! The CPU device 1D memory copy task.
        template<typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyCpu<DimInt<1u>, TViewDst, TViewSrc, TExtent>
            : public TaskCopyCpuBase<DimInt<1u>, TViewDst, TViewSrc, TExtent>
        {
            //-----------------------------------------------------------------------------
            using TaskCopyCpuBase<DimInt<1u>, TViewDst, TViewSrc, TExtent>::TaskCopyCpuBase;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator()() const -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#endif
                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    std::memcpy(
                        reinterpret_cast<void*>(this->m_dstMemNative),
                        reinterpret_cast<void const*>(this->m_srcMemNative),
                        static_cast<std::size_t>(this->m_extentWidthBytes));
                }
            }
        };
    } // namespace detail

    namespace traits
    {
        //#############################################################################
        //! The CPU device memory copy trait specialization.
        //!
        //! Copies from CPU memory into CPU memory.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevCpu, DevCpu>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent) -> alpaka::detail::TaskCopyCpu<TDim, TViewDst, TViewSrc, TExtent>
            {
                return alpaka::detail::TaskCopyCpu<TDim, TViewDst, TViewSrc, TExtent>(viewDst, viewSrc, extent);
            }
        };
    } // namespace traits
} // namespace alpaka
