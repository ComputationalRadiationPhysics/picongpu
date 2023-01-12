/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner, Andrea Bocci, Jan Stephan, Bernhard
 * Manfred Gruber, Antonio Di Pilato
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#    if _OPENACC < 201306
#        error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC 2.0 or higher!
#    endif

#    include <alpaka/core/AlignedAlloc.hpp>
#    include <alpaka/core/Assert.hpp>
#    include <alpaka/core/Vectorize.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/dev/DevOacc.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/mem/view/Traits.hpp>
#    include <alpaka/meta/Integral.hpp>
#    include <alpaka/meta/NdLoop.hpp>
#    include <alpaka/queue/QueueOaccBlocking.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <set>
#    include <tuple>
#    include <utility>

namespace alpaka
{
    namespace oacc
    {
        namespace detail
        {
            template<
                template<
                    typename TTDim,
                    typename TTViewDst,
                    typename TTViewSrc,
                    typename TTExtent,
                    typename TTCopyPred>
                class TTask,
                typename TDim,
                typename TViewDstFwd,
                typename TViewSrc,
                typename TExtent,
                typename TCopyPred>
            auto makeTaskCopyOacc(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                DevOacc dev,
                TCopyPred copyPred)
            {
                return TTask<TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent, TCopyPred>(
                    std::forward<TViewDstFwd>(viewDst),
                    viewSrc,
                    extent,
                    std::move(dev),
                    copyPred);
            }

            //! The OpenACC Nd device memory copy task.
            //!
            template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent, typename TCopyPred>
            struct TaskCopyOacc
            {
                using ExtentSize = alpaka::Idx<TExtent>;
                using DstSize = alpaka::Idx<TViewDst>;
                using SrcSize = alpaka::Idx<TViewSrc>;
                using Elem = alpaka::Elem<TViewSrc>;

                using Idx = alpaka::Idx<TExtent>;

                template<typename TViewDstFwd>
                ALPAKA_FN_HOST TaskCopyOacc(
                    TViewDstFwd&& viewDst,
                    TViewSrc const& viewSrc,
                    TExtent const& extent,
                    DevOacc dev,
                    TCopyPred copyPred)
                    : m_dev(std::move(dev))
                    , m_extent(getExtentVec(extent))
                    , m_extentWidthBytes(m_extent[TDim::value - 1u] * static_cast<ExtentSize>(sizeof(Elem)))
                    , m_dstPitchBytes(getPitchBytesVec(viewDst))
                    , m_srcPitchBytes(getPitchBytesVec(viewSrc))
                    ,
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    m_dstExtent(getExtentVec(viewDst))
                    , m_srcExtent(getExtentVec(viewSrc))
                    ,
#    endif
                    m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(viewDst)))
                    , m_srcMemNative(reinterpret_cast<std::uint8_t const*>(getPtrNative(viewSrc)))
                    , m_copyPred(copyPred)
                {
                    ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).foldrAll(std::logical_or<bool>()));
                    ALPAKA_ASSERT((castVec<SrcSize>(m_extent) <= m_srcExtent).foldrAll(std::logical_or<bool>()));
                    ALPAKA_ASSERT(static_cast<DstSize>(m_extentWidthBytes) <= m_dstPitchBytes[TDim::value - 1u]);
                    ALPAKA_ASSERT(static_cast<SrcSize>(m_extentWidthBytes) <= m_srcPitchBytes[TDim::value - 1u]);
                }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                ALPAKA_FN_HOST auto printDebug() const -> void
                {
                    std::cout << __func__ << " dev: " << m_dev.getNativeHandle() << " ew: " << m_extent
                              << " dw: " << m_dstExtent << " dptr: " << static_cast<void const*>(m_dstMemNative)
                              << " sw: " << m_srcExtent << " sptr: " << static_cast<void const*>(m_srcMemNative)
                              << std::endl;
                }
#    endif

                ALPAKA_FN_HOST auto operator()() const -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    printDebug();
#    endif
                    if constexpr(TDim::value == 1)
                    {
                        if(m_extent.prod() != 0)
                        {
                            m_dev.makeCurrent();
                            m_copyPred(
                                reinterpret_cast<void*>(m_dstMemNative),
                                const_cast<void*>(reinterpret_cast<void const*>(m_srcMemNative)),
                                static_cast<std::size_t>(m_extentWidthBytes));
                        }
                    }
                    else
                    {
                        using DimMin1 = DimInt<TDim::value - 1u>;

                        Vec<DimMin1, ExtentSize> const extentWithoutInnermost(subVecBegin<DimMin1>(m_extent));
                        // [z, y, x] -> [y, x] because the z pitch (the full size of the buffer) is not required.
                        Vec<DimMin1, DstSize> const dstPitchBytesWithoutOutmost(subVecEnd<DimMin1>(m_dstPitchBytes));
                        Vec<DimMin1, SrcSize> const srcPitchBytesWithoutOutmost(subVecEnd<DimMin1>(m_srcPitchBytes));

                        if(static_cast<std::size_t>(m_extent.prod()) != 0u)
                        {
                            m_dev.makeCurrent();
                            meta::ndLoopIncIdx(
                                extentWithoutInnermost,
                                [&](Vec<DimMin1, ExtentSize> const& idx)
                                {
                                    m_copyPred(
                                        reinterpret_cast<void*>(
                                            m_dstMemNative
                                            + (castVec<DstSize>(idx) * dstPitchBytesWithoutOutmost).sum()),
                                        const_cast<void*>(reinterpret_cast<const void*>(
                                            m_srcMemNative
                                            + (castVec<SrcSize>(idx) * srcPitchBytesWithoutOutmost).sum())),
                                        static_cast<std::size_t>(m_extentWidthBytes));
                                });
                        }
                    }
                }

                const DevOacc m_dev;
                Vec<TDim, ExtentSize> m_extent;
                ExtentSize const m_extentWidthBytes;
                Vec<TDim, DstSize> m_dstPitchBytes;
                Vec<TDim, SrcSize> m_srcPitchBytes;
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                Vec<TDim, DstSize> const m_dstExtent;
                Vec<TDim, SrcSize> const m_srcExtent;
#    endif
                std::uint8_t* const m_dstMemNative;
                std::uint8_t const* const m_srcMemNative;
                TCopyPred m_copyPred;
            };

            //! The OpenACC scalar memory copy task.
            template<typename TViewDst, typename TViewSrc, typename TExtent, typename TCopyPred>
            struct TaskCopyOacc<DimInt<0u>, TViewDst, TViewSrc, TExtent, TCopyPred>
            {
                using Elem = alpaka::Elem<TViewSrc>;
                using Idx = alpaka::Idx<TExtent>;

                template<typename TViewDstFwd>
                ALPAKA_FN_HOST TaskCopyOacc(
                    TViewDstFwd&& viewDst,
                    TViewSrc const& viewSrc,
                    TExtent const& /* extent */,
                    DevOacc const& dev,
                    TCopyPred copyPred)
                    : m_dev(dev)
                    , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(viewDst)))
                    , m_srcMemNative(reinterpret_cast<std::uint8_t const*>(getPtrNative(viewSrc)))
                    , m_copyPred(copyPred)
                {
                }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                ALPAKA_FN_HOST auto printDebug() const -> void
                {
                    std::cout << __func__ << " dev: " << m_dev.getNativeHandle() << " ew: " << Idx(1u)
                              << " dw: " << Idx(1u) << " dptr: " << static_cast<void const*>(m_dstMemNative)
                              << " sw: " << Idx(1u) << " sptr: " << static_cast<void const*>(m_srcMemNative)
                              << std::endl;
                }
#    endif

                ALPAKA_FN_HOST auto operator()() const -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    printDebug();
#    endif
                    m_dev.makeCurrent();
                    m_copyPred(
                        reinterpret_cast<void*>(this->m_dstMemNative),
                        const_cast<void*>(reinterpret_cast<void const*>(this->m_srcMemNative)),
                        sizeof(Elem));
                }

                const DevOacc m_dev;
                std::uint8_t* const m_dstMemNative;
                std::uint8_t const* const m_srcMemNative;
                TCopyPred m_copyPred;
            };

        } // namespace detail
    } // namespace oacc

    // Trait specializations for CreateTaskCopy.
    namespace trait
    {
        //! The CPU to OpenACC memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevOacc, DevCpu>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto devDst = getDev(viewDst);
                return alpaka::oacc::detail::makeTaskCopyOacc<alpaka::oacc::detail::TaskCopyOacc, TDim>(
                    std::forward<TViewDstFwd>(viewDst),
                    viewSrc,
                    extent,
                    std::move(devDst),
                    acc_memcpy_to_device);
            }
        };

        //! The OpenACC to CPU memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevCpu, DevOacc>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::oacc::detail::makeTaskCopyOacc<alpaka::oacc::detail::TaskCopyOacc, TDim>(
                    std::forward<TViewDstFwd>(viewDst),
                    viewSrc,
                    extent,
                    getDev(viewSrc),
                    acc_memcpy_from_device);
            }
        };

        //! The OpenACC to OpenACC memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevOacc, DevOacc>
        {
            template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDstFwd&& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto devDst = getDev(viewDst);
#    if _OPENACC >= 201510 && (!defined __GNUC__)
                // acc_memcpy_device is only available since OpenACC2.5, but we want the tests to compile anyway
                if(devDst.getNativeHandle() == getDev(viewSrc).getNativeHandle())
                {
                    return alpaka::oacc::detail::makeTaskCopyOacc<alpaka::oacc::detail::TaskCopyOacc, TDim>(
                        std::forward<TViewDstFwd>(viewDst),
                        viewSrc,
                        extent,
                        std::move(devDst),
                        acc_memcpy_device);
                }
                else
#    endif
                {
                    return alpaka::oacc::detail::makeTaskCopyOacc<alpaka::oacc::detail::TaskCopyOacc, TDim>(
                        std::forward<TViewDstFwd>(viewDst),
                        viewSrc,
                        extent,
                        devDst,
                        [devSrc = getDev(viewSrc), devDst](void* dst, void* src, std::size_t size)
                        {
                            auto deleter
                                = [](void* ptr) { core::alignedFree(core::vectorization::defaultAlignment, ptr); };
                            std::unique_ptr<void, decltype(deleter)> buf(
                                core::alignedAlloc(core::vectorization::defaultAlignment, size),
                                deleter);
                            devSrc.makeCurrent();
                            acc_memcpy_from_device(buf.get(), src, size);
                            devDst.makeCurrent();
                            acc_memcpy_to_device(dst, buf.get(), size);
                        });
                }
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
