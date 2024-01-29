/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan, Bernhard
 * Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/meta/Integral.hpp"
#include "alpaka/meta/NdLoop.hpp"

#include <cstring>

namespace alpaka
{
    class DevCpu;
} // namespace alpaka

namespace alpaka
{
    namespace detail
    {
        //! The CPU device memory copy task base.
        //!
        //! Copies from CPU memory into CPU memory.
        template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyCpuBase
        {
            static_assert(TDim::value > 0);

            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TViewDst>;
            using SrcSize = Idx<TViewSrc>;
            using Elem = alpaka::Elem<TViewSrc>;

            template<typename TViewFwd>
            TaskCopyCpuBase(TViewFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
                : m_extent(getExtents(extent))
                , m_extentWidthBytes(m_extent.back() * static_cast<ExtentSize>(sizeof(Elem)))
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                , m_dstExtent(getExtents(viewDst))
                , m_srcExtent(getExtents(viewSrc))
#endif
                , m_dstPitchBytes(getPitchesInBytes(viewDst))
                , m_srcPitchBytes(getPitchesInBytes(viewSrc))
                , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<std::uint8_t const*>(getPtrNative(viewSrc)))
            {
                if constexpr(TDim::value > 0)
                {
                    ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).all());
                    ALPAKA_ASSERT((castVec<SrcSize>(m_extent) <= m_srcExtent).all());
                    if constexpr(TDim::value > 1)
                    {
                        ALPAKA_ASSERT(static_cast<DstSize>(m_extentWidthBytes) <= m_dstPitchBytes[TDim::value - 2]);
                        ALPAKA_ASSERT(static_cast<SrcSize>(m_extentWidthBytes) <= m_srcPitchBytes[TDim::value - 2]);
                    }
                }
            }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
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

        //! The CPU device ND memory copy task.
        template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyCpu : public TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>
        {
            using DimMin1 = DimInt<TDim::value - 1u>;
            using typename TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>::ExtentSize;
            using typename TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>::DstSize;
            using typename TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>::SrcSize;

            using TaskCopyCpuBase<TDim, TViewDst, TViewSrc, TExtent>::TaskCopyCpuBase;

            ALPAKA_FN_HOST auto operator()() const -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#endif
                // [z, y, x] -> [z, y] because all elements with the innermost x dimension are handled within one
                // iteration.
                Vec<DimMin1, ExtentSize> const extentWithoutInnermost = subVecBegin<DimMin1>(this->m_extent);
                Vec<DimMin1, DstSize> const dstPitchBytesWithoutInnermost
                    = subVecBegin<DimMin1>(this->m_dstPitchBytes);
                Vec<DimMin1, SrcSize> const srcPitchBytesWithoutInnermost
                    = subVecBegin<DimMin1>(this->m_srcPitchBytes);

                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    meta::ndLoopIncIdx(
                        extentWithoutInnermost,
                        [&](Vec<DimMin1, ExtentSize> const& idx)
                        {
                            std::memcpy(
                                this->m_dstMemNative + (castVec<DstSize>(idx) * dstPitchBytesWithoutInnermost).sum(),
                                this->m_srcMemNative + (castVec<SrcSize>(idx) * srcPitchBytesWithoutInnermost).sum(),
                                static_cast<std::size_t>(this->m_extentWidthBytes));
                        });
                }
            }
        };

        //! The CPU device 1D memory copy task.
        template<typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyCpu<DimInt<1u>, TViewDst, TViewSrc, TExtent>
            : TaskCopyCpuBase<DimInt<1u>, TViewDst, TViewSrc, TExtent>
        {
            using TaskCopyCpuBase<DimInt<1u>, TViewDst, TViewSrc, TExtent>::TaskCopyCpuBase;

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

        //! The CPU device scalar memory copy task.
        //!
        //! Copies from CPU memory into CPU memory.
        template<typename TViewDst, typename TViewSrc, typename TExtent>
        struct TaskCopyCpu<DimInt<0u>, TViewDst, TViewSrc, TExtent>
        {
            using Elem = alpaka::Elem<TViewSrc>;

            template<typename TViewDstFwd>
            TaskCopyCpu(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, [[maybe_unused]] TExtent const& extent)
                : m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(viewDst)))
                , m_srcMemNative(reinterpret_cast<std::uint8_t const*>(getPtrNative(viewSrc)))
            {
                // all zero-sized extents are equivalent
                ALPAKA_ASSERT(getExtents(extent).prod() == 1u);
                ALPAKA_ASSERT(getExtents(viewDst).prod() == 1u);
                ALPAKA_ASSERT(getExtents(viewSrc).prod() == 1u);
            }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                using Scalar = Vec<DimInt<0u>, Idx<TExtent>>;
                std::cout << __func__ << " e: " << Scalar() << " ewb: " << sizeof(Elem) << " de: " << Scalar()
                          << " dptr: " << reinterpret_cast<void*>(m_dstMemNative) << " dpitchb: " << Scalar()
                          << " se: " << Scalar() << " sptr: " << reinterpret_cast<void const*>(m_srcMemNative)
                          << " spitchb: " << Scalar() << std::endl;
            }
#endif

            ALPAKA_FN_HOST auto operator()() const noexcept(ALPAKA_DEBUG < ALPAKA_DEBUG_FULL) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#endif
                std::memcpy(
                    reinterpret_cast<void*>(m_dstMemNative),
                    reinterpret_cast<void const*>(m_srcMemNative),
                    sizeof(Elem));
            }

            std::uint8_t* const m_dstMemNative;
            std::uint8_t const* const m_srcMemNative;
        };
    } // namespace detail

    namespace trait
    {
        //! The CPU device memory copy trait specialization.
        //!
        //! Copies from CPU memory into CPU memory.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevCpu, DevCpu>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
                -> alpaka::detail::TaskCopyCpu<TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
            {
                return {std::forward<TViewDstFwd>(viewDst), viewSrc, extent};
            }
        };
    } // namespace trait
} // namespace alpaka
