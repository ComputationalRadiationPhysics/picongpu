/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Debug.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/buf/sycl/Common.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/queue/QueueGenericSyclBlocking.hpp"
#include "alpaka/queue/QueueGenericSyclNonBlocking.hpp"
#include "alpaka/queue/Traits.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <type_traits>


#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka
{

    namespace detail
    {

        template<typename TDim, typename TView, typename TExtent, typename TValue>
        struct TaskFillSyclBase
        {
            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            template<typename TViewFwd>
            TaskFillSyclBase(TViewFwd&& view, TValue const& value, TExtent const& extent)
                : m_value(value)
                , m_extent(getExtents(extent))
                , m_extentWidth(m_extent.back())
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                , m_dstExtent(getExtents(view))
#    endif
                , m_dstPitchBytes(getPitchesInBytes(view))
                , m_dstMemNative(getPtrNative(view))
            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).all());
                if constexpr(TDim::value > 1)
                    ALPAKA_ASSERT(
                        m_extentWidth * static_cast<ExtentSize>(sizeof(Elem)) <= m_dstPitchBytes[TDim::value - 2]);
            }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            auto printDebug() const -> void
            {
                std::cout << __func__ << " e: " << m_extent << " ew: " << m_extentWidth << " de: " << m_dstExtent
                          << " dptr: " << reinterpret_cast<void*>(m_dstMemNative) << " dpitchb: " << m_dstPitchBytes
                          << std::endl;
            }
#    endif

            TValue const m_value;
            Vec<TDim, ExtentSize> const m_extent;
            ExtentSize const m_extentWidth;
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            Vec<TDim, DstSize> const m_dstExtent;
#    endif
            Vec<TDim, DstSize> const m_dstPitchBytes;
            Elem* const m_dstMemNative;

            static constexpr auto is_sycl_task = true;
        };

        template<typename TDim, typename TView, typename TExtent, typename TValue>
        struct TaskFillSycl : public TaskFillSyclBase<TDim, TView, TExtent, TValue>
        {
            using Base = TaskFillSyclBase<TDim, TView, TExtent, TValue>;
            using Base::Base;
            using typename Base::DstSize;
            using typename Base::ExtentSize;
            using DimMin1 = DimInt<TDim::value - 1u>;

            auto operator()(sycl::queue& queue, std::vector<sycl::event> const& requirements) const -> sycl::event
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#    endif
                Vec<DimMin1, ExtentSize> const extentWithoutInnermost(subVecBegin<DimMin1>(this->m_extent));
                Vec<DimMin1, DstSize> const dstPitchBytesWithoutInnermost(subVecBegin<DimMin1>(this->m_dstPitchBytes));

                std::vector<sycl::event> events;
                events.reserve(static_cast<std::size_t>(extentWithoutInnermost.prod()));

                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    using Elem = std::remove_cvref_t<decltype(this->m_value)>;

                    meta::ndLoopIncIdx(
                        extentWithoutInnermost,
                        [&](Vec<DimMin1, ExtentSize> const& idx)
                        {
                            auto offsetBytes = (castVec<DstSize>(idx) * dstPitchBytesWithoutInnermost).sum();
                            Elem* ptr = reinterpret_cast<Elem*>(
                                reinterpret_cast<std::uint8_t*>(this->m_dstMemNative) + offsetBytes);

                            assert(this->m_extentWidth >= 0);

                            events.push_back(queue.fill<TValue>(
                                ptr,
                                this->m_value,
                                static_cast<std::size_t>(this->m_extentWidth),
                                requirements));
                        });
                }


                return queue.ext_oneapi_submit_barrier(events);
            }
        };

        template<typename TView, typename TExtent, typename TValue>
        struct TaskFillSycl<DimInt<1u>, TView, TExtent, TValue>
            : public TaskFillSyclBase<DimInt<1u>, TView, TExtent, TValue>
        {
            using Base = TaskFillSyclBase<DimInt<1u>, TView, TExtent, TValue>;
            using Base::Base;

            auto operator()(sycl::queue& queue, std::vector<sycl::event> const& requirements) const -> sycl::event
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#    endif
                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    return queue.fill(
                        this->m_dstMemNative,
                        this->m_value,
                        static_cast<std::size_t>(this->m_extentWidth),
                        requirements);
                }
                else
                {
                    return queue.ext_oneapi_submit_barrier();
                }
            }
        };

        template<typename TView, typename TExtent, typename TValue>
        struct TaskFillSycl<DimInt<0u>, TView, TExtent, TValue>
        {
            using Elem = alpaka::Elem<TView>;

            template<typename TViewFwd>
            TaskFillSycl(TViewFwd&& view, TValue const& value, [[maybe_unused]] TExtent const& extent)
                : m_value(value)
                , m_dstMemNative(getPtrNative(view))
            {
                ALPAKA_ASSERT(getExtents(extent).prod() == 1u);
                ALPAKA_ASSERT(getExtents(view).prod() == 1u);
            }

            auto operator()(sycl::queue& queue, std::vector<sycl::event> const& requirements) const -> sycl::event
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                return queue.fill(m_dstMemNative, m_value, 1, requirements);
            }

            TValue const m_value;
            Elem* const m_dstMemNative;
            static constexpr auto is_sycl_task = true;
        };

    } // namespace detail

    namespace trait
    {
        template<typename TDim, typename TPlatform>
        struct CreateTaskFill<TDim, DevGenericSycl<TPlatform>>
        {
            template<typename TExtent, typename TView, typename TValue>
            static auto createTaskFill(TView& view, TValue const& value, TExtent const& extent)
                -> alpaka::detail::TaskFillSycl<TDim, TView, TExtent, TValue>
            {
                return alpaka::detail::TaskFillSycl<TDim, TView, TExtent, TValue>(view, value, extent);
            }
        };
    } // namespace trait

} // namespace alpaka
#endif
