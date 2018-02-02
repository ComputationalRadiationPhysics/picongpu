/**
 * \file
 * Copyright 2014-2017 Benjamin Worpitz
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

#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/IsIntegralSuperset.hpp>

#include <cassert>
#include <cstring>

namespace alpaka
{
    namespace dev
    {
        class DevCpu;
    }
}

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace cpu
            {
                namespace detail
                {
                    //#############################################################################
                    //! The CPU device memory copy task base.
                    //!
                    //! Copies from CPU memory into CPU memory.
                    template<
                        typename TDim,
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyBase
                    {
                        using ExtentSize = size::Size<TExtent>;
                        using DstSize = size::Size<TViewDst>;
                        using SrcSize = size::Size<TViewSrc>;
                        using Elem = elem::Elem<TViewSrc>;

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == TDim::value,
                            "The destination view and the input TDim are required to have the same dimensionality!");

                        static_assert(
                            meta::IsIntegralSuperset<DstSize, ExtentSize>::value,
                            "The destination view and the extent are required to have compatible size type!");
                        static_assert(
                            meta::IsIntegralSuperset<SrcSize, ExtentSize>::value,
                            "The source view and the extent are required to have compatible size type!");

                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        //-----------------------------------------------------------------------------
                        TaskCopyBase(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent) :
                                m_extent(extent::getExtentVec(extent)),
                                m_extentWidthBytes(m_extent[TDim::value - 1u] * static_cast<ExtentSize>(sizeof(Elem))),
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                                m_dstExtent(extent::getExtentVec(viewDst)),
                                m_srcExtent(extent::getExtentVec(viewSrc)),
#endif
                                m_dstPitchBytes(mem::view::getPitchBytesVec(viewDst)),
                                m_srcPitchBytes(mem::view::getPitchBytesVec(viewSrc)),

                                m_dstMemNative(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<std::uint8_t const *>(mem::view::getPtrNative(viewSrc)))
                        {
                            assert((vec::cast<DstSize>(m_extent) <= m_dstExtent).foldrAll(std::logical_or<bool>()));
                            assert((vec::cast<SrcSize>(m_extent) <= m_srcExtent).foldrAll(std::logical_or<bool>()));
                            assert(static_cast<DstSize>(m_extentWidthBytes) <= m_dstPitchBytes[TDim::value - 1u]);
                            assert(static_cast<SrcSize>(m_extentWidthBytes) <= m_srcPitchBytes[TDim::value - 1u]);
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " e: " << m_extent
                                << " ewb: " << this->m_extentWidthBytes
                                << " de: " << m_dstExtent
                                << " dptr: " << reinterpret_cast<void *>(m_dstMemNative)
                                << " dpitchb: " << m_dstPitchBytes
                                << " se: " << m_srcExtent
                                << " sptr: " << reinterpret_cast<void const *>(m_srcMemNative)
                                << " spitchb: " << m_srcPitchBytes
                                << std::endl;
                        }
#endif

                        vec::Vec<TDim, ExtentSize> const m_extent;
                        ExtentSize const m_extentWidthBytes;
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                        vec::Vec<TDim, DstSize> const m_dstExtent;
                        vec::Vec<TDim, SrcSize> const m_srcExtent;
#endif
                        vec::Vec<TDim, DstSize> const m_dstPitchBytes;
                        vec::Vec<TDim, SrcSize> const m_srcPitchBytes;

                        std::uint8_t * const m_dstMemNative;
                        std::uint8_t const * const m_srcMemNative;
                    };



                    //#############################################################################
                    //! The CPU device ND memory copy task.
                    template<
                        typename TDim,
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopy : public TaskCopyBase<TDim, TViewDst, TViewSrc, TExtent>
                    {
                        using DimMin1 = dim::DimInt<TDim::value - 1u>;

                        //-----------------------------------------------------------------------------
                        using TaskCopyBase<TDim, TViewDst, TViewSrc, TExtent>::TaskCopyBase;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto operator()() const
                        -> void
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            this->printDebug();
#endif
                            // [z, y, x] -> [z, y] because all elements with the innermost x dimension are handled within one iteration.
                            using ExtentSize = typename TaskCopyBase<TDim, TViewDst, TViewSrc, TExtent>::ExtentSize;
                            vec::Vec<DimMin1, ExtentSize> const extentWithoutInnermost(vec::subVecBegin<DimMin1>(this->m_extent));
                            // [z, y, x] -> [y, x] because the z pitch (the full size of the buffer) is not required.
                            using DstSize = typename TaskCopyBase<TDim, TViewDst, TViewSrc, TExtent>::DstSize;
                            vec::Vec<DimMin1, DstSize> const dstPitchBytesWithoutOutmost(vec::subVecEnd<DimMin1>(this->m_dstPitchBytes));
                            using SrcSize = typename TaskCopyBase<TDim, TViewDst, TViewSrc, TExtent>::SrcSize;
                            vec::Vec<DimMin1, SrcSize> const srcPitchBytesWithoutOutmost(vec::subVecEnd<DimMin1>(this->m_srcPitchBytes));

                            if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                            {
                                meta::ndLoopIncIdx(
                                    extentWithoutInnermost,
                                    [&](vec::Vec<DimMin1, ExtentSize> const & idx)
                                    {
                                        std::memcpy(
                                            reinterpret_cast<void *>(this->m_dstMemNative + (vec::cast<DstSize>(idx) * dstPitchBytesWithoutOutmost).foldrAll(std::plus<DstSize>())),
                                            reinterpret_cast<void const *>(this->m_srcMemNative + (vec::cast<SrcSize>(idx) * srcPitchBytesWithoutOutmost).foldrAll(std::plus<SrcSize>())),
                                            static_cast<std::size_t>(this->m_extentWidthBytes));
                                    });
                            }
                        }
                    };

                    //#############################################################################
                    //! The CPU device 1D memory copy task.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopy<
                        dim::DimInt<1u>,
                        TViewDst,
                        TViewSrc,
                        TExtent> : public TaskCopyBase<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>
                    {
                        //-----------------------------------------------------------------------------
                        using TaskCopyBase<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>::TaskCopyBase;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto operator()() const
                        -> void
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            this->printDebug();
#endif
                            if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                            {
                                std::memcpy(
                                    reinterpret_cast<void *>(this->m_dstMemNative),
                                    reinterpret_cast<void const *>(this->m_srcMemNative),
                                    static_cast<std::size_t>(this->m_extentWidthBytes));
                            }
                        }
                    };
                }
            }

            namespace traits
            {
                //#############################################################################
                //! The CPU device memory copy trait specialization.
                //!
                //! Copies from CPU memory into CPU memory.
                template<
                    typename TDim>
                struct TaskCopy<
                    TDim,
                    dev::DevCpu,
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
                    -> cpu::detail::TaskCopy<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        return
                            cpu::detail::TaskCopy<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent);
                    }
                };
            }
        }
    }
}
