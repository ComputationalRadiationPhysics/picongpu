/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
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

    namespace detail
    {
        //! The CPU device ND memory set task base.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskSetCpuBase
        {
            static_assert(TDim::value > 0);

            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            template<typename TViewFwd>
            TaskSetCpuBase(TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent)
                : m_byte(byte)
                , m_extent(getExtents(extent))
                , m_extentWidthBytes(m_extent.back() * static_cast<ExtentSize>(sizeof(Elem)))
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                , m_dstExtent(getExtents(view))
#endif
                , m_dstPitchBytes(getPitchesInBytes(view))
                , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(view)))
            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).all());
                if constexpr(TDim::value > 1)
                    ALPAKA_ASSERT(m_extentWidthBytes <= m_dstPitchBytes[TDim::value - 2]);
            }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " e: " << this->m_extent << " ewb: " << this->m_extentWidthBytes
                          << " de: " << this->m_dstExtent << " dptr: " << reinterpret_cast<void*>(this->m_dstMemNative)
                          << " dpitchb: " << this->m_dstPitchBytes << std::endl;
            }
#endif

            std::uint8_t const m_byte;
            Vec<TDim, ExtentSize> const m_extent;
            ExtentSize const m_extentWidthBytes;
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            Vec<TDim, DstSize> const m_dstExtent;
#endif
            Vec<TDim, DstSize> const m_dstPitchBytes;
            std::uint8_t* const m_dstMemNative;
        };

        //! The CPU device ND memory set task.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskSetCpu : public TaskSetCpuBase<TDim, TView, TExtent>
        {
            using DimMin1 = DimInt<TDim::value - 1u>;
            using typename TaskSetCpuBase<TDim, TView, TExtent>::ExtentSize;
            using typename TaskSetCpuBase<TDim, TView, TExtent>::DstSize;

            using TaskSetCpuBase<TDim, TView, TExtent>::TaskSetCpuBase;

            ALPAKA_FN_HOST auto operator()() const -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#endif
                // [z, y, x] -> [z, y] because all elements with the innermost x dimension are handled within one
                // iteration.
                Vec<DimMin1, ExtentSize> const extentWithoutInnermost = subVecBegin<DimMin1>(this->m_extent);
                Vec<DimMin1, DstSize> const dstPitchBytesWithoutOutmost = subVecBegin<DimMin1>(this->m_dstPitchBytes);

                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    meta::ndLoopIncIdx(
                        extentWithoutInnermost,
                        [&](Vec<DimMin1, ExtentSize> const& idx)
                        {
                            std::memset(
                                this->m_dstMemNative + (castVec<DstSize>(idx) * dstPitchBytesWithoutOutmost).sum(),
                                this->m_byte,
                                static_cast<std::size_t>(this->m_extentWidthBytes));
                        });
                }
            }
        };

        //! The CPU device 1D memory set task.
        template<typename TView, typename TExtent>
        struct TaskSetCpu<DimInt<1u>, TView, TExtent> : public TaskSetCpuBase<DimInt<1u>, TView, TExtent>
        {
            using TaskSetCpuBase<DimInt<1u>, TView, TExtent>::TaskSetCpuBase;

            ALPAKA_FN_HOST auto operator()() const -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#endif
                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    std::memset(
                        this->m_dstMemNative,
                        this->m_byte,
                        static_cast<std::size_t>(this->m_extentWidthBytes));
                }
            }
        };

        //! The CPU device scalar memory set task.
        template<typename TView, typename TExtent>
        struct TaskSetCpu<DimInt<0u>, TView, TExtent>
        {
            using ExtentSize = Idx<TExtent>;
            using Scalar = Vec<DimInt<0u>, ExtentSize>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            template<typename TViewFwd>
            TaskSetCpu(TViewFwd&& view, std::uint8_t const& byte, [[maybe_unused]] TExtent const& extent)
                : m_byte(byte)
                , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(view)))
            {
                // all zero-sized extents are equivalent
                ALPAKA_ASSERT(getExtents(extent).prod() == 1u);
                ALPAKA_ASSERT(getExtents(view).prod() == 1u);
            }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " e: " << Scalar() << " ewb: " << sizeof(Elem) << " de: " << Scalar()
                          << " dptr: " << reinterpret_cast<void*>(m_dstMemNative) << " dpitchb: " << Scalar()
                          << std::endl;
            }
#endif

            ALPAKA_FN_HOST auto operator()() const noexcept(ALPAKA_DEBUG < ALPAKA_DEBUG_FULL) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#endif
                std::memset(m_dstMemNative, m_byte, sizeof(Elem));
            }

            std::uint8_t const m_byte;
            std::uint8_t* const m_dstMemNative;
        };
    } // namespace detail

    namespace trait
    {
        //! The CPU device memory set trait specialization.
        template<typename TDim>
        struct CreateTaskMemset<TDim, DevCpu>
        {
            template<typename TExtent, typename TViewFwd>
            ALPAKA_FN_HOST static auto createTaskMemset(
                TViewFwd&& view,
                std::uint8_t const& byte,
                TExtent const& extent) -> alpaka::detail::TaskSetCpu<TDim, std::remove_reference_t<TViewFwd>, TExtent>
            {
                return {std::forward<TViewFwd>(view), byte, extent};
            }
        };
    } // namespace trait
} // namespace alpaka
