/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/dim/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/size/Traits.hpp>

#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Common.hpp>

#include <type_traits>
#include <cassert>

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
                typename TSize>
            class ViewSubView
            {
            public:
                using Dev = TDev;
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! \param view The view this view is a sub-view of.
                template<
                    typename TView>
                ViewSubView(
                    TView const & view) :
                        m_viewParentView(
                            mem::view::getPtrNative(view),
                            dev::getDev(view),
                            extent::getExtentVecEnd<TDim>(view),
                            mem::view::getPitchBytesVecEnd<TDim>(view)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(view)),
                        m_offsetsElements(vec::Vec<TDim, TSize>::all(0))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;
                }
                //-----------------------------------------------------------------------------
                //! \param view The view this view is a sub-view of.
                template<
                    typename TView>
                ViewSubView(
                    TView & view) :
                        m_viewParentView(
                            mem::view::getPtrNative(view),
                            dev::getDev(view),
                            extent::getExtentVecEnd<TDim>(view),
                            mem::view::getPitchBytesVecEnd<TDim>(view)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(view)),
                        m_offsetsElements(vec::Vec<TDim, TSize>::all(0))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TSize, size::Size<TView>>::value,
                        "The size type of TView and the TSize template parameter have to be identical!");
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
                    TView const & view,
                    TExtent const & extentElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_viewParentView(
                            mem::view::getPtrNative(view),
                            dev::getDev(view),
                            extent::getExtentVecEnd<TDim>(view),
                            mem::view::getPitchBytesVecEnd<TDim>(view)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extentElements)),
                        m_offsetsElements(offset::getOffsetVecEnd<TDim>(relativeOffsetsElements))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtent>>::value,
                        "The buffer and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<TSize, size::Size<TExtent>>::value,
                        "The size type of TExtent and the TSize template parameter have to be identical!");
                    static_assert(
                        std::is_same<TSize, size::Size<TView>>::value,
                        "The size type of TView and the TSize template parameter have to be identical!");

                    assert((offset::getOffsetX(relativeOffsetsElements)+extent::getWidth(extentElements)) <= extent::getWidth(view));
                    assert((offset::getOffsetY(relativeOffsetsElements)+extent::getHeight(extentElements)) <= extent::getHeight(view));
                    assert((offset::getOffsetZ(relativeOffsetsElements)+extent::getDepth(extentElements)) <= extent::getDepth(view));
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
                            extent::getExtentVecEnd<TDim>(view),
                            mem::view::getPitchBytesVecEnd<TDim>(view)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extentElements)),
                        m_offsetsElements(offset::getOffsetVecEnd<TDim>(relativeOffsetsElements))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtent>>::value,
                        "The buffer and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<TSize, size::Size<TExtent>>::value,
                        "The size type of TExtent and the TSize template parameter have to be identical!");
                    static_assert(
                        std::is_same<TSize, size::Size<TView>>::value,
                        "The size type of TView and the TSize template parameter have to be identical!");

                    assert((offset::getOffsetX(relativeOffsetsElements)+extent::getWidth(extentElements)) <= extent::getWidth(view));
                    assert((offset::getOffsetY(relativeOffsetsElements)+extent::getHeight(extentElements)) <= extent::getHeight(view));
                    assert((offset::getOffsetZ(relativeOffsetsElements)+extent::getDepth(extentElements)) <= extent::getDepth(view));
                }

            public:
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize> m_viewParentView; // This wraps the parent view.
                vec::Vec<TDim, TSize> m_extentElements;     // The extent of this view.
                vec::Vec<TDim, TSize> m_offsetsElements;    // The offset relative to the parent view.
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
                typename TSize>
            struct DevType<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The ViewSubView device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct GetDev<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & view)
                -> TDev
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
                typename TSize>
            struct DimType<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
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
                typename TSize>
            struct ElemType<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
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
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct GetExtent<
                TIdx,
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & extent)
                -> TSize
                {
                    return extent.m_extentElements[TIdx::value];
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
                //#############################################################################
                //! The ViewSubView native pointer get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev,
                    typename TSize>
                struct GetPtrNative<
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
                {
                private:
                    using IdxSequence = meta::MakeIntegerSequence<std::size_t, TDim::value>;
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & view)
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
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> & view)
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
                        meta::IntegerSequence<std::size_t, TIndices...> const &)
                    -> TSize
                    {
                        return
                            meta::foldr(
                                std::plus<TSize>(),
                                pitchedOffsetBytesDim<TIndices>(view)...);
                    }
                    //-----------------------------------------------------------------------------
                    template<
                        std::size_t Tidx,
                        typename TView>
                    ALPAKA_FN_HOST static auto pitchedOffsetBytesDim(
                        TView const & view)
                    -> TSize
                    {
                        return
                            offset::getOffset<Tidx>(view)
                            * mem::view::getPitchBytes<Tidx + 1u>(view);
                    }
                };

                //#############################################################################
                //! The ViewSubView pitch get trait specialization.
                template<
                    typename TIdx,
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPitchBytes<
                    TIdx,
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & view)
                    -> TSize
                    {
                        return
                            mem::view::getPitchBytes<TIdx::value>(
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
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct GetOffset<
                TIdx,
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & offset)
                -> TSize
                {
                    return offset.m_offsetsElements[TIdx::value];
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView size type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct SizeType<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
