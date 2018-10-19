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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/elem/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/stream/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/meta/Fold.hpp>
#include <alpaka/core/Common.hpp>

#include <boost/config.hpp>
#include <boost/core/ignore_unused.hpp>

#include <iosfwd>

namespace alpaka
{
    namespace mem
    {
        //-----------------------------------------------------------------------------
        //! The view specifics.
        namespace view
        {
            //-----------------------------------------------------------------------------
            //! The view traits.
            namespace traits
            {
                //#############################################################################
                //! The native pointer get trait.
                template<
                    typename TView,
                    typename TSfinae = void>
                struct GetPtrNative;

                //#############################################################################
                //! The pointer on device get trait.
                template<
                    typename TView,
                    typename TDev,
                    typename TSfinae = void>
                struct GetPtrDev;

                namespace detail
                {
                    //#############################################################################
                    template<
                        typename TIdx,
                        typename TView,
                        typename TSfinae = void>
                    struct GetPitchBytesDefault;
                }

                //#############################################################################
                //! The pitch in bytes.
                //! This is the distance in bytes in the linear memory between two consecutive elements in the next higher dimension (TIdx-1).
                //!
                //! The default implementation uses the extent to calculate the pitch.
                template<
                    typename TIdx,
                    typename TView,
                    typename TSfinae = void>
                struct GetPitchBytes
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        TView const & view)
                    -> size::Size<TView>
                    {
                        return detail::GetPitchBytesDefault<TIdx, TView>::getPitchBytesDefault(view);
                    }
                };

                namespace detail
                {
                    //#############################################################################
                    template<
                        typename TIdx,
                        typename TView>
                    struct GetPitchBytesDefault<
                        TIdx,
                        TView,
                        typename std::enable_if<TIdx::value < (dim::Dim<TView>::value - 1)>::type>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST static auto getPitchBytesDefault(
                            TView const & view)
                        -> size::Size<TView>
                        {
                            return
                                extent::getExtent<TIdx::value>(view)
                                * GetPitchBytes<dim::DimInt<TIdx::value+1>, TView>::getPitchBytes(view);
                        }
                    };
                    //#############################################################################
                    template<
                        typename TView>
                    struct GetPitchBytesDefault<
                        dim::DimInt<dim::Dim<TView>::value - 1u>,
                        TView>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST static auto getPitchBytesDefault(
                            TView const & view)
                        -> size::Size<TView>
                        {
                            return
                                extent::getExtent<dim::Dim<TView>::value - 1u>(view)
                                * sizeof(elem::Elem<TView>);
                        }
                    };
                    //#############################################################################
                    template<
                        typename TView>
                    struct GetPitchBytesDefault<
                        dim::DimInt<dim::Dim<TView>::value>,
                        TView>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST static auto getPitchBytesDefault(
                            TView const &)
                        -> size::Size<TView>
                        {
                            return
                                sizeof(elem::Elem<TView>);
                        }
                    };
                }

                //#############################################################################
                //! The memory set trait.
                //!
                //! Fills the view with data.
                template<
                    typename TDim,
                    typename TDev,
                    typename TSfinae = void>
                struct TaskSet;

                //#############################################################################
                //! The memory copy trait.
                //!
                //! Copies memory from one view into another view possibly on a different device.
                template<
                    typename TDim,
                    typename TDevDst,
                    typename TDevSrc,
                    typename TSfinae = void>
                struct TaskCopy;

                //#############################################################################
                //! The static device memory view creation trait.
                template<
                    typename TDev,
                    typename TSfinae = void>
                struct CreateStaticDevMemView;
            }

            //-----------------------------------------------------------------------------
            //! Gets the native pointer of the memory view.
            //!
            //! \param view The memory view.
            //! \return The native pointer.
            template<
                typename TView>
            ALPAKA_FN_HOST auto getPtrNative(
                TView const & view)
            -> elem::Elem<TView> const *
            {
                return
                    traits::GetPtrNative<
                        TView>
                    ::getPtrNative(
                        view);
            }
            //-----------------------------------------------------------------------------
            //! Gets the native pointer of the memory view.
            //!
            //! \param view The memory view.
            //! \return The native pointer.
            template<
                typename TView>
            ALPAKA_FN_HOST auto getPtrNative(
                TView & view)
            -> elem::Elem<TView> *
            {
                return
                    traits::GetPtrNative<
                        TView>
                    ::getPtrNative(
                        view);
            }

            //-----------------------------------------------------------------------------
            //! Gets the pointer to the view on the given device.
            //!
            //! \param view The memory view.
            //! \param dev The device.
            //! \return The pointer on the device.
            template<
                typename TView,
                typename TDev>
            ALPAKA_FN_HOST auto getPtrDev(
                TView const & view,
                TDev const & dev)
            -> elem::Elem<TView> const *
            {
                return
                    traits::GetPtrDev<
                        TView,
                        TDev>
                    ::getPtrDev(
                        view,
                        dev);
            }
            //-----------------------------------------------------------------------------
            //! Gets the pointer to the view on the given device.
            //!
            //! \param view The memory view.
            //! \param dev The device.
            //! \return The pointer on the device.
            template<
                typename TView,
                typename TDev>
            ALPAKA_FN_HOST auto getPtrDev(
                TView & view,
                TDev const & dev)
            -> elem::Elem<TView> *
            {
                return
                    traits::GetPtrDev<
                        TView,
                        TDev>
                    ::getPtrDev(
                        view,
                        dev);
            }

            //-----------------------------------------------------------------------------
            //! \return The pitch in bytes. This is the distance in bytes between two consecutive elements in the given dimension.
            template<
                std::size_t Tidx,
                typename TView>
            ALPAKA_FN_HOST auto getPitchBytes(
                TView const & view)
            -> size::Size<TView>
            {
                return
                    traits::GetPitchBytes<
                        dim::DimInt<Tidx>,
                        TView>
                    ::getPitchBytes(
                        view);
            }

            //-----------------------------------------------------------------------------
            //! Create a memory set task.
            //!
            //! \param view The memory view to fill.
            //! \param byte Value to set for each element of the specified view.
            //! \param extent The extent of the view to fill.
            template<
                typename TExtent,
                typename TView>
            ALPAKA_FN_HOST auto taskSet(
                TView & view,
                std::uint8_t const & byte,
                TExtent const & extent)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                traits::TaskSet<
                    dim::Dim<TView>,
                    dev::Dev<TView>>
                ::taskSet(
                    view,
                    byte,
                    extent))
#endif
            {
                static_assert(
                    dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                    "The view and the extent are required to have the same dimensionality!");

                return
                    traits::TaskSet<
                        dim::Dim<TView>,
                        dev::Dev<TView>>
                    ::taskSet(
                        view,
                        byte,
                        extent);
            }

            //-----------------------------------------------------------------------------
            //! Sets the memory to the given value asynchronously.
            //!
            //! \param view The memory view to fill.
            //! \param byte Value to set for each element of the specified view.
            //! \param extent The extent of the view to fill.
            //! \param stream The stream to enqueue the view fill task into.
            template<
                typename TExtent,
                typename TView,
                typename TStream>
            ALPAKA_FN_HOST auto set(
                TStream & stream,
                TView & view,
                std::uint8_t const & byte,
                TExtent const & extent)
            -> void
            {
                stream::enqueue(
                    stream,
                    mem::view::taskSet(
                        view,
                        byte,
                        extent));
            }

            //-----------------------------------------------------------------------------
            //! Creates a memory copy task.
            //!
            //! \param viewDst The destination memory view.
            //! \param viewSrc The source memory view.
            //! \param extent The extent of the view to copy.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            ALPAKA_FN_HOST auto taskCopy(
                TViewDst & viewDst,
                TViewSrc const & viewSrc,
                TExtent const & extent)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                traits::TaskCopy<
                    dim::Dim<TViewDst>,
                    dev::Dev<TViewDst>,
                    dev::Dev<TViewSrc>>
                ::taskCopy(
                    viewDst,
                    viewSrc,
                    extent))
#endif
            {
                static_assert(
                    dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                    "The source and the destination view are required to have the same dimensionality!");
                static_assert(
                    dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                    "The destination view and the extent are required to have the same dimensionality!");
                static_assert(
                    std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                    "The source and the destination view are required to have the same element type!");

                return
                    traits::TaskCopy<
                        dim::Dim<TViewDst>,
                        dev::Dev<TViewDst>,
                        dev::Dev<TViewSrc>>
                    ::taskCopy(
                        viewDst,
                        viewSrc,
                        extent);
            }

            //-----------------------------------------------------------------------------
            //! Copies memory possibly between different memory spaces asynchronously.
            //!
            //! \param viewDst The destination memory view.
            //! \param viewSrc The source memory view.
            //! \param extent The extent of the view to copy.
            //! \param stream The stream to enqueue the view copy task into.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst,
                typename TStream>
            ALPAKA_FN_HOST auto copy(
                TStream & stream,
                TViewDst & viewDst,
                TViewSrc const & viewSrc,
                TExtent const & extent)
            -> void
            {
                stream::enqueue(
                    stream,
                    mem::view::taskCopy(
                        viewDst,
                        viewSrc,
                        extent));
            }

            namespace detail
            {
                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename TView>
                struct Print
                {
                    ALPAKA_FN_HOST static auto print(
                        TView const & view,
                        elem::Elem<TView> const * const ptr,
                        vec::Vec<dim::Dim<TView>, size::Size<TView>> const & extent,
                        std::ostream & os,
                        std::string const & elementSeparator,
                        std::string const & rowSeparator,
                        std::string const & rowPrefix,
                        std::string const & rowSuffix)
                    -> void
                    {
                        os << rowPrefix;

                        auto const pitch(view::getPitchBytes<TDim::value+1u>(view));
                        auto const lastIdx(extent[TDim::value]-1u);
                        for(auto i(decltype(lastIdx)(0)); i<=lastIdx ;++i)
                        {
                            Print<
                                dim::DimInt<TDim::value+1u>,
                                TView>
                            ::print(
                                view,
                                reinterpret_cast<elem::Elem<TView> const *>(reinterpret_cast<std::uint8_t const *>(ptr)+i*pitch),
                                extent,
                                os,
                                elementSeparator,
                                rowSeparator,
                                rowPrefix,
                                rowSuffix);

                            // While we are not at the end of a row, add the row separator.
                            if(i != lastIdx)
                            {
                                os << rowSeparator;
                            }
                        }

                        os << rowSuffix;
                    }
                };
                //-----------------------------------------------------------------------------
                template<
                    typename TView>
                struct Print<
                    dim::DimInt<dim::Dim<TView>::value-1u>,
                    TView>
                {
                    ALPAKA_FN_HOST static auto print(
                        TView const & view,
                        elem::Elem<TView> const * const ptr,
                        vec::Vec<dim::Dim<TView>, size::Size<TView>> const & extent,
                        std::ostream & os,
                        std::string const & elementSeparator,
                        std::string const & rowSeparator,
                        std::string const & rowPrefix,
                        std::string const & rowSuffix)
                    -> void
                    {
                        boost::ignore_unused(view);
                        boost::ignore_unused(rowSeparator);

                        os << rowPrefix;

                        auto const lastIdx(extent[dim::Dim<TView>::value-1u]-1u);
                        for(auto i(decltype(lastIdx)(0)); i<=lastIdx ;++i)
                        {
                            // Add the current element.
                            os << *(ptr+i);

                            // While we are not at the end of a line, add the element separator.
                            if(i != lastIdx)
                            {
                                os << elementSeparator;
                            }
                        }

                        os << rowSuffix;
                    }
                };
            }
            //-----------------------------------------------------------------------------
            //! Prints the content of the view to the given stream.
            // \TODO: Add precision flag.
            // \TODO: Add column alignment flag.
            template<
                typename TView>
            ALPAKA_FN_HOST auto print(
                TView const & view,
                std::ostream & os,
                std::string const & elementSeparator = ", ",
                std::string const & rowSeparator = "\n",
                std::string const & rowPrefix = "[",
                std::string const & rowSuffix = "]")
            -> void
            {
                detail::Print<
                    dim::DimInt<0u>,
                    TView>
                ::print(
                    view,
                    mem::view::getPtrNative(view),
                    extent::getExtentVec(view),
                    os,
                    elementSeparator,
                    rowSeparator,
                    rowPrefix,
                    rowSuffix);
            }

            namespace detail
            {
                //#############################################################################
                //! A class with a create method that returns the pitch for each index.
                template<
                    std::size_t Tidx>
                struct CreatePitchBytes
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TPitch>
                    ALPAKA_FN_HOST static auto create(
                        TPitch const & pitch)
                    -> size::Size<TPitch>
                    {
                        return mem::view::getPitchBytes<Tidx>(pitch);
                    }
                };
            }
            //-----------------------------------------------------------------------------
            //! \return The pitch vector.
            template<
                typename TPitch>
            ALPAKA_FN_HOST auto getPitchBytesVec(
                TPitch const & pitch = TPitch())
            -> vec::Vec<dim::Dim<TPitch>, size::Size<TPitch>>
            {
                return
                    vec::createVecFromIndexedFnWorkaround<
                        dim::Dim<TPitch>,
                        size::Size<TPitch>,
                        detail::CreatePitchBytes>(
                            pitch);
            }
            //-----------------------------------------------------------------------------
            //! \return The pitch but only the last N elements.
            template<
                typename TDim,
                typename TPitch>
            ALPAKA_FN_HOST auto getPitchBytesVecEnd(
                TPitch const & pitch = TPitch())
            -> vec::Vec<TDim, size::Size<TPitch>>
            {
                using IdxOffset = std::integral_constant<std::intmax_t, static_cast<std::intmax_t>(dim::Dim<TPitch>::value) - static_cast<std::intmax_t>(TDim::value)>;
                return
                    vec::createVecFromIndexedFnOffsetWorkaround<
                        TDim,
                        size::Size<TPitch>,
                        detail::CreatePitchBytes,
                        IdxOffset>(
                            pitch);
            }

            //-----------------------------------------------------------------------------
            //! \return A view to static device memory.
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TElem,
                typename TDev,
                typename TExtent>
            ALPAKA_FN_HOST_ACC auto createStaticDevMemView(
                TElem * pMem,
                TDev const & dev,
                TExtent const & extent)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                traits::CreateStaticDevMemView<
                        TDev>
                    ::createStaticDevMemView(
                        pMem,
                        dev,
                        extent))
#endif
            {
                return
                    traits::CreateStaticDevMemView<
                        TDev>
                    ::createStaticDevMemView(
                        pMem,
                        dev,
                        extent);
            }
        }
    }
}
