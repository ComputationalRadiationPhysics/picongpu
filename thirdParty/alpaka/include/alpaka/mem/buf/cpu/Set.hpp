/**
 * \file
 * Copyright 2014-2017 Benjamin Worpitz, Erik Zenker
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
                    //! The CPU device ND memory set task base.
                    template<
                        typename TDim,
                        typename TView,
                        typename TExtent>
                    struct TaskSetBase
                    {
                        using ExtentSize = size::Size<TExtent>;
                        using DstSize = size::Size<TView>;
                        using Elem = elem::Elem<TView>;

                        static_assert(
                            dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                            "The destination view and the extent are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TView>::value == TDim::value,
                            "The destination view and the input TDim are required to have the same dimensionality!");

                        static_assert(
                            meta::IsIntegralSuperset<DstSize, ExtentSize>::value,
                            "The view and the extent are required to have compatible size type!");

                        //-----------------------------------------------------------------------------
                        TaskSetBase(
                            TView & view,
                            std::uint8_t const & byte,
                            TExtent const & extent) :
                                m_byte(byte),
                                m_extent(extent::getExtentVec(extent)),
                                m_extentWidthBytes(m_extent[TDim::value - 1u] * static_cast<ExtentSize>(sizeof(Elem))),
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                                m_dstExtent(extent::getExtentVec(view)),
#endif
                                m_dstPitchBytes(mem::view::getPitchBytesVec(view)),
                                m_dstMemNative(reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(view)))
                        {
                            assert((vec::cast<DstSize>(m_extent) <= m_dstExtent).foldrAll(std::logical_or<bool>()));
                            assert(m_extentWidthBytes <= m_dstPitchBytes[TDim::value - 1u]);
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " e: " << this->m_extent
                                << " ewb: " << this->m_extentWidthBytes
                                << " de: " << this->m_dstExtent
                                << " dptr: " << reinterpret_cast<void *>(this->m_dstMemNative)
                                << " dpitchb: " << this->m_dstPitchBytes
                                << std::endl;
                        }
#endif

                        std::uint8_t const m_byte;
                        vec::Vec<TDim, ExtentSize> const m_extent;
                        ExtentSize const m_extentWidthBytes;
#if (!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                        vec::Vec<TDim, DstSize> const m_dstExtent;
#endif
                        vec::Vec<TDim, DstSize> const m_dstPitchBytes;
                        std::uint8_t * const m_dstMemNative;
                    };

                    //#############################################################################
                    //! The CPU device ND memory set task.
                    template<
                        typename TDim,
                        typename TView,
                        typename TExtent>
                    struct TaskSet : public TaskSetBase<TDim, TView, TExtent>
                    {
                        using DimMin1 = dim::DimInt<TDim::value - 1u>;

                        //-----------------------------------------------------------------------------
                        using TaskSetBase<TDim, TView, TExtent>::TaskSetBase;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto operator()() const
                        -> void
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            this->printDebug();
#endif
                            // [z, y, x] -> [z, y] because all elements with the innermost x dimension are handled within one iteration.
                            using ExtentSize = typename TaskSetBase<TDim, TView, TExtent>::ExtentSize;
                            vec::Vec<DimMin1, ExtentSize> const extentWithoutInnermost(vec::subVecBegin<DimMin1>(this->m_extent));
                            // [z, y, x] -> [y, x] because the z pitch (the full size of the buffer) is not required.
                            using DstSize = typename TaskSetBase<TDim, TView, TExtent>::DstSize;
                            vec::Vec<DimMin1, DstSize> const dstPitchBytesWithoutOutmost(vec::subVecEnd<DimMin1>(this->m_dstPitchBytes));

                            if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                            {
                                meta::ndLoopIncIdx(
                                    extentWithoutInnermost,
                                    [&](vec::Vec<DimMin1, ExtentSize> const & idx)
                                    {
                                        std::memset(
                                            reinterpret_cast<void *>(this->m_dstMemNative + (vec::cast<DstSize>(idx) * dstPitchBytesWithoutOutmost).foldrAll(std::plus<DstSize>())),
                                            this->m_byte,
                                            static_cast<std::size_t>(this->m_extentWidthBytes));
                                    });
                            }
                        }
                    };

                    //#############################################################################
                    //! The CPU device 1D memory set task.
                    template<
                        typename TView,
                        typename TExtent>
                    struct TaskSet<
                        dim::DimInt<1u>,
                        TView,
                        TExtent> : public TaskSetBase<dim::DimInt<1u>, TView, TExtent>
                    {
                        //-----------------------------------------------------------------------------
                        using TaskSetBase<dim::DimInt<1u>, TView, TExtent>::TaskSetBase;

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
                                std::memset(
                                    reinterpret_cast<void *>(this->m_dstMemNative),
                                    this->m_byte,
                                    static_cast<std::size_t>(this->m_extentWidthBytes));
                            }
                        }
                    };
                }
            }

            namespace traits
            {
                //#############################################################################
                //! The CPU device memory set trait specialization.
                template<
                    typename TDim>
                struct TaskSet<
                    TDim,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TView>
                    ALPAKA_FN_HOST static auto taskSet(
                        TView & view,
                        std::uint8_t const & byte,
                        TExtent const & extent)
                    -> cpu::detail::TaskSet<
                        TDim,
                        TView,
                        TExtent>
                    {
                        return
                            cpu::detail::TaskSet<
                                TDim,
                                TView,
                                TExtent>(
                                    view,
                                    byte,
                                    extent);
                    }
                };
            }
        }
    }
}
