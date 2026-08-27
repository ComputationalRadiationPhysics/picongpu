/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/meta/Integral.hpp"
#include "alpaka/meta/NdLoop.hpp"

namespace alpaka
{
    class DevCpu;

    namespace detail
    {
        //! The CPU device N-dimensional memory fill task.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskFillCpu
        {
            static_assert(TDim::value > 0);

            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            static_assert(std::is_trivially_copyable_v<Elem>, "Only trivially copyable types supported for fill");

            template<typename TViewFwd>
            TaskFillCpu(TViewFwd&& view, Elem const& value, TExtent const& extent)
                : m_value(value)
                , m_extent(getExtents(extent))
#if(!defined(NDEBUG))
                , m_dstExtent(getExtents(view))
#endif
                , m_dstPitchBytes(getPitchesInBytes(view))
                , m_dstMemNative(getPtrNative(view))
            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).all());
                if constexpr(TDim::value > 0)
                {
                    ALPAKA_ASSERT(static_cast<std::size_t>(m_dstPitchBytes[TDim::value - 1]) >= sizeof(Elem));
                    ALPAKA_ASSERT(static_cast<std::size_t>(m_dstPitchBytes[TDim::value - 1]) % alignof(Elem) == 0);
                }
                if constexpr(TDim::value > 1)
                {
                    for(int dim = TDim::value - 2; dim >= 0; --dim)
                    {
                        ALPAKA_ASSERT(
                            static_cast<std::size_t>(m_dstPitchBytes[dim])
                            >= static_cast<std::size_t>(m_dstPitchBytes[dim + 1] * m_dstExtent[dim + 1]));
                        ALPAKA_ASSERT(static_cast<std::size_t>(m_dstPitchBytes[dim]) % alignof(Elem) == 0);
                    }
                }
                ALPAKA_ASSERT(reinterpret_cast<std::uintptr_t>(m_dstMemNative) % alignof(Elem) == 0);
            }

            ALPAKA_FN_HOST auto operator()() const -> void
            {
                if(static_cast<std::size_t>(m_extent.prod()) != 0u)
                {
                    meta::ndLoopIncIdx(
                        m_extent,
                        [&](Vec<TDim, ExtentSize> const& idx)
                        {
                            // All elements of m_dstPitchBytes are multiples of the alignment of Elem.
                            std::uintptr_t offsetBytes = static_cast<std::uintptr_t>((idx * m_dstPitchBytes).sum());
                            Elem* elem = reinterpret_cast<Elem*>(__builtin_assume_aligned(
                                reinterpret_cast<std::uint8_t*>(m_dstMemNative) + offsetBytes,
                                alignof(Elem)));
                            *elem = m_value;
                        });
                }
            }

        private:
            Elem const m_value;
            Vec<TDim, ExtentSize> const m_extent;
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            Vec<TDim, DstSize> const m_dstExtent;
#endif
            Vec<TDim, DstSize> const m_dstPitchBytes;
            Elem* const m_dstMemNative;
        };

        //! The CPU device 0-dimensional memory fill task specialisation.
        template<typename TView, typename TExtent>
        struct TaskFillCpu<DimInt<0u>, TView, TExtent>
        {
            using Elem = alpaka::Elem<TView>;

            template<typename TViewFwd>
            TaskFillCpu(TViewFwd&& view, Elem const& value, [[maybe_unused]] TExtent const& extent)
                : m_value(value)
                , m_dstMemNative(getPtrNative(view))
            {
                ALPAKA_ASSERT(getExtents(extent).prod() == 1u);
                ALPAKA_ASSERT(getExtents(view).prod() == 1u);
                ALPAKA_ASSERT(reinterpret_cast<std::uintptr_t>(m_dstMemNative) % alignof(Elem) == 0);
            }

            ALPAKA_FN_HOST auto operator()() const noexcept -> void
            {
                *m_dstMemNative = m_value;
            }

        private:
            Elem const m_value;
            Elem* const m_dstMemNative;
        };
    } // namespace detail

    namespace trait
    {
        //! The memory fill task trait specialization for CPU devices.
        template<typename TDim>
        struct CreateTaskFill<TDim, DevCpu>
        {
            template<typename TExtent, typename TViewFwd>
            ALPAKA_FN_HOST static auto createTaskFill(
                TViewFwd&& view,
                alpaka::Elem<std::remove_reference_t<TViewFwd>> const& value,
                TExtent const& extent)
            {
                using TView = std::remove_reference_t<TViewFwd>;
                using Elem = alpaka::Elem<TView>;
                static_assert(
                    std::is_trivially_copyable_v<Elem>,
                    "Only trivially copyable types are supported for fill");

                return alpaka::detail::TaskFillCpu<TDim, TView, TExtent>{std::forward<TViewFwd>(view), value, extent};
            }
        };
    } // namespace trait

} // namespace alpaka
