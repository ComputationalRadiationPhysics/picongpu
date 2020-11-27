/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner
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

    namespace detail
    {
        //#############################################################################
        //! The CPU device ND memory set task base.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskSetCpuBase
        {
            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            static_assert(!std::is_const<TView>::value, "The destination view can not be const!");

            static_assert(
                Dim<TView>::value == Dim<TExtent>::value,
                "The destination view and the extent are required to have the same dimensionality!");
            static_assert(
                Dim<TView>::value == TDim::value,
                "The destination view and the input TDim are required to have the same dimensionality!");

            static_assert(
                meta::IsIntegralSuperset<DstSize, ExtentSize>::value,
                "The view and the extent are required to have compatible idx type!");

            //-----------------------------------------------------------------------------
            TaskSetCpuBase(TView& view, std::uint8_t const& byte, TExtent const& extent)
                : m_byte(byte)
                , m_extent(extent::getExtentVec(extent))
                , m_extentWidthBytes(m_extent[TDim::value - 1u] * static_cast<ExtentSize>(sizeof(Elem)))
                ,
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                m_dstExtent(extent::getExtentVec(view))
                ,
#endif
                m_dstPitchBytes(getPitchBytesVec(view))
                , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(view)))
            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).foldrAll(std::logical_or<bool>()));
                ALPAKA_ASSERT(m_extentWidthBytes <= m_dstPitchBytes[TDim::value - 1u]);
            }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            //-----------------------------------------------------------------------------
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

        //#############################################################################
        //! The CPU device ND memory set task.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskSetCpu : public TaskSetCpuBase<TDim, TView, TExtent>
        {
            using DimMin1 = DimInt<TDim::value - 1u>;
            using typename TaskSetCpuBase<TDim, TView, TExtent>::ExtentSize;
            using typename TaskSetCpuBase<TDim, TView, TExtent>::DstSize;

            //-----------------------------------------------------------------------------
            using TaskSetCpuBase<TDim, TView, TExtent>::TaskSetCpuBase;

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
                // [z, y, x] -> [y, x] because the z pitch (the full idx of the buffer) is not required.
                Vec<DimMin1, DstSize> const dstPitchBytesWithoutOutmost(subVecEnd<DimMin1>(this->m_dstPitchBytes));

                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    meta::ndLoopIncIdx(extentWithoutInnermost, [&](Vec<DimMin1, ExtentSize> const& idx) {
                        std::memset(
                            reinterpret_cast<void*>(
                                this->m_dstMemNative
                                + (castVec<DstSize>(idx) * dstPitchBytesWithoutOutmost)
                                      .foldrAll(std::plus<DstSize>())),
                            this->m_byte,
                            static_cast<std::size_t>(this->m_extentWidthBytes));
                    });
                }
            }
        };

        //#############################################################################
        //! The CPU device 1D memory set task.
        template<typename TView, typename TExtent>
        struct TaskSetCpu<DimInt<1u>, TView, TExtent> : public TaskSetCpuBase<DimInt<1u>, TView, TExtent>
        {
            //-----------------------------------------------------------------------------
            using TaskSetCpuBase<DimInt<1u>, TView, TExtent>::TaskSetCpuBase;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator()() const -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#endif
                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    std::memset(
                        reinterpret_cast<void*>(this->m_dstMemNative),
                        this->m_byte,
                        static_cast<std::size_t>(this->m_extentWidthBytes));
                }
            }
        };
    } // namespace detail

    namespace traits
    {
        //#############################################################################
        //! The CPU device memory set trait specialization.
        template<typename TDim>
        struct CreateTaskMemset<TDim, DevCpu>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(TView& view, std::uint8_t const& byte, TExtent const& extent)
                -> alpaka::detail::TaskSetCpu<TDim, TView, TExtent>
            {
                return alpaka::detail::TaskSetCpu<TDim, TView, TExtent>(view, byte, extent);
            }
        };
    } // namespace traits
} // namespace alpaka
