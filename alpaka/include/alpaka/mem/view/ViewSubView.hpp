/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dim/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/vec/Vec.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Common.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            //#############################################################################
            //! A sub-view to a view.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TIdx>
            class ViewSubView
            {
                static_assert(
                    !std::is_const<TIdx>::value,
                    "The idx type of the view can not be const!");

                using Dev = alpaka::dev::Dev<TDev>;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param view The view this view is a sub-view of.
                //! \param extentElements The extent in elements.
                //! \param relativeOffsetsElements The offsets in elements.
                template<
                    typename TView,
                    typename TOffsets,
                    typename TExtent>
                ViewSubView(
                    TView const & view,
                    TExtent const & extentElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_viewParentView(
                            mem::view::getPtrNative(view),
                            dev::getDev(view),
                            extent::getExtentVec(view),
                            mem::view::getPitchBytesVec(view)),
                        m_extentElements(extent::getExtentVec(extentElements)),
                        m_offsetsElements(offset::getOffsetVec(relativeOffsetsElements))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<Dev, dev::Dev<TView>>::value,
                        "The dev type of TView and the Dev template parameter have to be identical!");

                    static_assert(
                        std::is_same<TIdx, idx::Idx<TView>>::value,
                        "The idx type of TView and the TIdx template parameter have to be identical!");
                    static_assert(
                        std::is_same<TIdx, idx::Idx<TExtent>>::value,
                        "The idx type of TExtent and the TIdx template parameter have to be identical!");
                    static_assert(
                        std::is_same<TIdx, idx::Idx<TOffsets>>::value,
                        "The idx type of TOffsets and the TIdx template parameter have to be identical!");

                    static_assert(
                        std::is_same<TDim, dim::Dim<TView>>::value,
                        "The dim type of TView and the TDim template parameter have to be identical!");
                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtent>>::value,
                        "The dim type of TExtent and the TDim template parameter have to be identical!");
                    static_assert(
                        std::is_same<TDim, dim::Dim<TOffsets>>::value,
                        "The dim type of TOffsets and the TDim template parameter have to be identical!");

                    ALPAKA_ASSERT(((m_offsetsElements + m_extentElements) <= extent::getExtentVec(view)).foldrAll(std::logical_and<bool>()));
                }
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param view The view this view is a sub-view of.
                //! \param extentElements The extent in elements.
                //! \param relativeOffsetsElements The offsets in elements.
                template<
                    typename TView,
                    typename TOffsets,
                    typename TExtent>
                ViewSubView(
                    TView & view,
                    TExtent const & extentElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_viewParentView(
                            mem::view::getPtrNative(view),
                            dev::getDev(view),
                            extent::getExtentVec(view),
                            mem::view::getPitchBytesVec(view)),
                        m_extentElements(extent::getExtentVec(extentElements)),
                        m_offsetsElements(offset::getOffsetVec(relativeOffsetsElements))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<Dev, dev::Dev<TView>>::value,
                        "The dev type of TView and the Dev template parameter have to be identical!");

                    static_assert(
                        std::is_same<TIdx, idx::Idx<TView>>::value,
                        "The idx type of TView and the TIdx template parameter have to be identical!");
                    static_assert(
                        std::is_same<TIdx, idx::Idx<TExtent>>::value,
                        "The idx type of TExtent and the TIdx template parameter have to be identical!");
                    static_assert(
                        std::is_same<TIdx, idx::Idx<TOffsets>>::value,
                        "The idx type of TOffsets and the TIdx template parameter have to be identical!");

                    static_assert(
                        std::is_same<TDim, dim::Dim<TView>>::value,
                        "The dim type of TView and the TDim template parameter have to be identical!");
                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtent>>::value,
                        "The dim type of TExtent and the TDim template parameter have to be identical!");
                    static_assert(
                        std::is_same<TDim, dim::Dim<TOffsets>>::value,
                        "The dim type of TOffsets and the TDim template parameter have to be identical!");

                    ALPAKA_ASSERT(((m_offsetsElements + m_extentElements) <= extent::getExtentVec(view)).foldrAll(std::logical_and<bool>()));
                }

                //-----------------------------------------------------------------------------
                //! \param view The view this view is a sub-view of.
                template<
                    typename TView>
                explicit ViewSubView(
                    TView const & view) :
                        ViewSubView(
                            view,
                            view,
                            vec::Vec<TDim, TIdx>::all(0))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;
                }

                //-----------------------------------------------------------------------------
                //! \param view The view this view is a sub-view of.
                template<
                    typename TView>
                explicit ViewSubView(
                    TView & view) :
                        ViewSubView(
                            view,
                            view,
                            vec::Vec<TDim, TIdx>::all(0))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;
                }

            public:
                mem::view::ViewPlainPtr<Dev, TElem, TDim, TIdx> m_viewParentView; // This wraps the parent view.
                vec::Vec<TDim, TIdx> m_extentElements;     // The extent of this view.
                vec::Vec<TDim, TIdx> m_offsetsElements;    // The offset relative to the parent view.
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for ViewSubView.
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView device type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TIdx>
            struct DevType<
                mem::view::ViewSubView<TDev, TElem, TDim, TIdx>>
            {
                using type = alpaka::dev::Dev<TDev>;
            };

            //#############################################################################
            //! The ViewSubView device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TIdx>
            struct GetDev<
                mem::view::ViewSubView<TDev, TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    mem::view::ViewSubView<TDev, TElem, TDim, TIdx> const & view)
                -> alpaka::dev::Dev<TDev>
                {
                    return
                        dev::getDev(
                            view.m_viewParentView);
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView dimension getter trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TIdx>
            struct DimType<
                mem::view::ViewSubView<TDev, TElem, TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView memory element type get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TIdx>
            struct ElemType<
                mem::view::ViewSubView<TDev, TElem, TDim, TIdx>>
            {
                using type = TElem;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView width get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TDev,
                typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                mem::view::ViewSubView<TDev, TElem, TDim, TIdx>,
                std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::view::ViewSubView<TDev, TElem, TDim, TIdx> const & extent)
                -> TIdx
                {
                    return extent.m_extentElements[TIdxIntegralConst::value];
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wcast-align" // "cast from 'std::uint8_t*' to 'TElem*' increases required alignment of target type"
#endif
                //#############################################################################
                //! The ViewSubView native pointer get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev,
                    typename TIdx>
                struct GetPtrNative<
                    mem::view::ViewSubView<TDev, TElem, TDim, TIdx>>
                {
                private:
                    using IdxSequence = std::make_index_sequence<TDim::value>;
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::view::ViewSubView<TDev, TElem, TDim, TIdx> const & view)
                    -> TElem const *
                    {
                        // \TODO: pre-calculate this pointer for faster execution.
                        return
                            reinterpret_cast<TElem const *>(
                                reinterpret_cast<std::uint8_t const *>(mem::view::getPtrNative(view.m_viewParentView))
                                + pitchedOffsetBytes(view, IdxSequence()));
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::view::ViewSubView<TDev, TElem, TDim, TIdx> & view)
                    -> TElem *
                    {
                        // \TODO: pre-calculate this pointer for faster execution.
                        return
                            reinterpret_cast<TElem *>(
                                reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(view.m_viewParentView))
                                + pitchedOffsetBytes(view, IdxSequence()));
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! For a 3D vector this calculates:
                    //!
                    //! offset::getOffset<0u>(view) * mem::view::getPitchBytes<1u>(view)
                    //! + offset::getOffset<1u>(view) * mem::view::getPitchBytes<2u>(view)
                    //! + offset::getOffset<2u>(view) * mem::view::getPitchBytes<3u>(view)
                    //! while mem::view::getPitchBytes<3u>(view) is equivalent to sizeof(TElem)
                    template<
                        typename TView,
                        std::size_t... TIndices>
                    ALPAKA_FN_HOST static auto pitchedOffsetBytes(
                        TView const & view,
                        std::index_sequence<TIndices...> const &)
                    -> TIdx
                    {
                        return
                            meta::foldr(
                                std::plus<TIdx>(),
                                pitchedOffsetBytesDim<TIndices>(view)...);
                    }
                    //-----------------------------------------------------------------------------
                    template<
                        std::size_t Tidx,
                        typename TView>
                    ALPAKA_FN_HOST static auto pitchedOffsetBytesDim(
                        TView const & view)
                    -> TIdx
                    {
                        return
                            offset::getOffset<Tidx>(view)
                            * mem::view::getPitchBytes<Tidx + 1u>(view);
                    }
                };
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif

                //#############################################################################
                //! The ViewSubView pitch get trait specialization.
                template<
                    typename TIdxIntegralConst,
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPitchBytes<
                    TIdxIntegralConst,
                    mem::view::ViewSubView<TDev, TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::view::ViewSubView<TDev, TElem, TDim, TIdx> const & view)
                    -> TIdx
                    {
                        return
                            mem::view::getPitchBytes<TIdxIntegralConst::value>(
                                view.m_viewParentView);
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView x offset get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TDev,
                typename TIdx>
            struct GetOffset<
                TIdxIntegralConst,
                mem::view::ViewSubView<TDev, TElem, TDim, TIdx>,
                std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                    mem::view::ViewSubView<TDev, TElem, TDim, TIdx> const & offset)
                -> TIdx
                {
                    return offset.m_offsetsElements[TIdxIntegralConst::value];
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView idx type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TIdx>
            struct IdxType<
                mem::view::ViewSubView<TDev, TElem, TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}
