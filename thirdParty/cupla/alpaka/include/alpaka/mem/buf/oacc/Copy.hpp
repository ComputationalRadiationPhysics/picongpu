/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner
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

#    include <alpaka/core/Assert.hpp>
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

#    if _OPENACC < 201510
#        include <vector>
#    endif

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
                typename TViewDst,
                typename TViewSrc,
                typename TExtent,
                typename TCopyPred>
            auto makeTaskCopyOacc(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent,
                DevOacc const& dev,
                TCopyPred copyPred)
            {
                return TTask<TDim, TViewDst, TViewSrc, TExtent, TCopyPred>(viewDst, viewSrc, extent, dev, copyPred);
            }

            //#############################################################################
            //! The OpenACC device memory copy task base.
            //!
            template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent, typename TCopyPred>
            struct TaskCopyOaccBase
            {
                using ExtentSize = alpaka::Idx<TExtent>;
                using DstSize = alpaka::Idx<TViewDst>;
                using SrcSize = alpaka::Idx<TViewSrc>;
                using Elem = alpaka::Elem<TViewSrc>;

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
                    std::is_same<alpaka::Elem<TViewDst>, typename std::remove_const<alpaka::Elem<TViewSrc>>::type>::
                        value,
                    "The source and the destination view are required to have the same element type!");

                using Idx = alpaka::Idx<TExtent>;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST TaskCopyOaccBase(
                    TViewDst& viewDst,
                    TViewSrc const& viewSrc,
                    TExtent const& extent,
                    DevOacc const& dev,
                    TCopyPred copyPred)
                    : m_dev(dev)
                    , m_extent(extent::getExtentVec(extent))
                    , m_extentWidthBytes(m_extent[TDim::value - 1u] * static_cast<ExtentSize>(sizeof(Elem)))
                    , m_dstPitchBytes(getPitchBytesVec(viewDst))
                    , m_srcPitchBytes(getPitchBytesVec(viewSrc))
                    ,
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                    m_dstExtent(extent::getExtentVec(viewDst))
                    , m_srcExtent(extent::getExtentVec(viewSrc))
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
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto printDebug() const -> void
                {
                    std::cout << __func__ << " dev: " << m_dev.iDevice() << " ew: " << m_extent
                              << " dw: " << m_dstExtent << " dptr: " << static_cast<const void*>(m_dstMemNative)
                              << " sw: " << m_srcExtent << " sptr: " << static_cast<const void*>(m_srcMemNative)
                              << std::endl;
                }
#    endif
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

            //#############################################################################
            //! The OpenACC Nd device memory copy task.
            //!
            template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent, typename TCopyPred>
            struct TaskCopyOacc : public TaskCopyOaccBase<TDim, TViewDst, TViewSrc, TExtent, TCopyPred>
            {
                using DimMin1 = DimInt<TDim::value - 1u>;
                using typename TaskCopyOaccBase<TDim, TViewDst, TViewSrc, TExtent, TCopyPred>::ExtentSize;
                using typename TaskCopyOaccBase<TDim, TViewDst, TViewSrc, TExtent, TCopyPred>::DstSize;
                using typename TaskCopyOaccBase<TDim, TViewDst, TViewSrc, TExtent, TCopyPred>::SrcSize;

                //-----------------------------------------------------------------------------
                using TaskCopyOaccBase<TDim, TViewDst, TViewSrc, TExtent, TCopyPred>::TaskCopyOaccBase;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator()() const -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    this->printDebug();
#    endif
                    Vec<DimMin1, ExtentSize> const extentWithoutInnermost(subVecBegin<DimMin1>(this->m_extent));
                    // [z, y, x] -> [y, x] because the z pitch (the full size of the buffer) is not required.
                    Vec<DimMin1, DstSize> const dstPitchBytesWithoutOutmost(subVecEnd<DimMin1>(this->m_dstPitchBytes));
                    Vec<DimMin1, SrcSize> const srcPitchBytesWithoutOutmost(subVecEnd<DimMin1>(this->m_srcPitchBytes));

                    if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                    {
                        this->m_dev.makeCurrent();
                        meta::ndLoopIncIdx(extentWithoutInnermost, [&](Vec<DimMin1, ExtentSize> const& idx) {
                            this->m_copyPred(
                                reinterpret_cast<void*>(
                                    this->m_dstMemNative
                                    + (castVec<DstSize>(idx) * dstPitchBytesWithoutOutmost).sum()),
                                const_cast<void*>(reinterpret_cast<const void*>(
                                    this->m_srcMemNative
                                    + (castVec<SrcSize>(idx) * srcPitchBytesWithoutOutmost).sum())),
                                static_cast<std::size_t>(this->m_extentWidthBytes));
                        });
                    }
                }
            };

            //#############################################################################
            //! The 1d OpenACC memory copy task.
            template<typename TViewDst, typename TViewSrc, typename TExtent, typename TCopyPred>
            struct TaskCopyOacc<DimInt<1>, TViewDst, TViewSrc, TExtent, TCopyPred>
                : public TaskCopyOaccBase<DimInt<1>, TViewDst, TViewSrc, TExtent, TCopyPred>
            {
                //-----------------------------------------------------------------------------
                using TaskCopyOaccBase<DimInt<1u>, TViewDst, TViewSrc, TExtent, TCopyPred>::TaskCopyOaccBase;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator()() const -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    this->printDebug();
#    endif
                    if(this->m_extent.prod() != 0)
                    {
                        this->m_dev.makeCurrent();
                        this->m_copyPred(
                            reinterpret_cast<void*>(this->m_dstMemNative),
                            const_cast<void*>(reinterpret_cast<void const*>(this->m_srcMemNative)),
                            static_cast<std::size_t>(this->m_extentWidthBytes));
                    }
                }
            };
        } // namespace detail
    } // namespace oacc

    //-----------------------------------------------------------------------------
    // Trait specializations for CreateTaskCopy.
    namespace traits
    {
        //#############################################################################
        //! The CPU to OpenACC memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevOacc, DevCpu>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::oacc::detail::
                    makeTaskCopyOacc<alpaka::oacc::detail::TaskCopyOacc, TDim, TViewDst, TViewSrc, TExtent>(
                        viewDst,
                        viewSrc,
                        extent,
                        getDev(viewDst),
                        acc_memcpy_to_device);
            }
        };

        //#############################################################################
        //! The OpenACC to CPU memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevCpu, DevOacc>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return alpaka::oacc::detail::
                    makeTaskCopyOacc<alpaka::oacc::detail::TaskCopyOacc, TDim, TViewDst, TViewSrc, TExtent>(
                        viewDst,
                        viewSrc,
                        extent,
                        getDev(viewSrc),
                        acc_memcpy_from_device);
            }
        };

        //#############################################################################
        //! The OpenACC to OpenACC memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevOacc, DevOacc>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(
                TViewDst& viewDst,
                TViewSrc const& viewSrc,
                TExtent const& extent)
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                ALPAKA_ASSERT(
                    getDev(viewDst).m_spDevOaccImpl->iDevice() == getDev(viewSrc).m_spDevOaccImpl->iDevice());

                return alpaka::oacc::detail::
                    makeTaskCopyOacc<alpaka::oacc::detail::TaskCopyOacc, TDim, TViewDst, TViewSrc, TExtent>(
                        viewDst,
                        viewSrc,
                        extent,
                        getDev(viewDst),
#    if _OPENACC >= 201510 && (!defined __GNUC__)
                        acc_memcpy_device
#    else
                        // acc_memcpy_device is only available since OpenACC2.5
                        // , but we want the tests to compile anyway
                        [](void* dst, void* src, std::size_t size) {
                            std::vector<std::size_t> buf(size / sizeof(std::size_t));
                            acc_memcpy_from_device(static_cast<void*>(buf.data()), src, size);
                            acc_memcpy_to_device(dst, static_cast<void*>(buf.data()), size);
                        }
#    endif
                    );
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
