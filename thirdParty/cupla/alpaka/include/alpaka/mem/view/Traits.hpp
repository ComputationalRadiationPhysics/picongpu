/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/elem/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/meta/Fold.hpp>
#include <alpaka/vec/Vec.hpp>

#include <iosfwd>
#include <type_traits>

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
                    -> idx::Idx<TView>
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
                        std::enable_if_t<TIdx::value < (dim::Dim<TView>::value - 1)>>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST static auto getPitchBytesDefault(
                            TView const & view)
                        -> idx::Idx<TView>
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
                        -> idx::Idx<TView>
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
                        -> idx::Idx<TView>
                        {
                            return
                                sizeof(elem::Elem<TView>);
                        }
                    };
                }

                //#############################################################################
                //! The memory set task trait.
                //!
                //! Fills the view with data.
                template<
                    typename TDim,
                    typename TDev,
                    typename TSfinae = void>
                struct CreateTaskSet;

                //#############################################################################
                //! The memory copy task trait.
                //!
                //! Copies memory from one view into another view possibly on a different device.
                template<
                    typename TDim,
                    typename TDevDst,
                    typename TDevSrc,
                    typename TSfinae = void>
                struct CreateTaskCopy;

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
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                std::size_t Tidx,
                typename TView>
            ALPAKA_FN_HOST_ACC
            auto getPitchBytes(
                TView const & view)
            -> idx::Idx<TView>
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
            ALPAKA_FN_HOST auto createTaskSet(
                TView & view,
                std::uint8_t const & byte,
                TExtent const & extent)
            {
                static_assert(
                    dim::Dim<TView>::value == dim::Dim<TExtent>::value,
                    "The view and the extent are required to have the same dimensionality!");

                return
                    traits::CreateTaskSet<
                        dim::Dim<TView>,
                        dev::Dev<TView>>
                    ::createTaskSet(
                        view,
                        byte,
                        extent);
            }

            //-----------------------------------------------------------------------------
            //! Sets the memory to the given value.
            //!
            //! \param queue The queue to enqueue the view fill task into.
            //! \param view The memory view to fill.
            //! \param byte Value to set for each element of the specified view.
            //! \param extent The extent of the view to fill.
            template<
                typename TExtent,
                typename TView,
                typename TQueue>
            ALPAKA_FN_HOST auto set(
                TQueue & queue,
                TView & view,
                std::uint8_t const & byte,
                TExtent const & extent)
            -> void
            {
                queue::enqueue(
                    queue,
                    mem::view::createTaskSet(
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
            ALPAKA_FN_HOST auto createTaskCopy(
                TViewDst & viewDst,
                TViewSrc const & viewSrc,
                TExtent const & extent)
            {
                static_assert(
                    dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                    "The source and the destination view are required to have the same dimensionality!");
                static_assert(
                    dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                    "The destination view and the extent are required to have the same dimensionality!");
                static_assert(
                    std::is_same<elem::Elem<TViewDst>, std::remove_const_t<elem::Elem<TViewSrc>>>::value,
                    "The source and the destination view are required to have the same element type!");

                return
                    traits::CreateTaskCopy<
                        dim::Dim<TViewDst>,
                        dev::Dev<TViewDst>,
                        dev::Dev<TViewSrc>>
                    ::createTaskCopy(
                        viewDst,
                        viewSrc,
                        extent);
            }

            //-----------------------------------------------------------------------------
            //! Copies memory possibly between different memory spaces.
            //!
            //! \param queue The queue to enqueue the view copy task into.
            //! \param viewDst The destination memory view.
            //! \param viewSrc The source memory view.
            //! \param extent The extent of the view to copy.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst,
                typename TQueue>
            ALPAKA_FN_HOST auto copy(
                TQueue & queue,
                TViewDst & viewDst,
                TViewSrc const & viewSrc,
                TExtent const & extent)
            -> void
            {
                queue::enqueue(
                    queue,
                    mem::view::createTaskCopy(
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
                        vec::Vec<dim::Dim<TView>, idx::Idx<TView>> const & extent,
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
                        vec::Vec<dim::Dim<TView>, idx::Idx<TView>> const & extent,
                        std::ostream & os,
                        std::string const & elementSeparator,
                        std::string const & rowSeparator,
                        std::string const & rowPrefix,
                        std::string const & rowSuffix)
                    -> void
                    {
                        alpaka::ignore_unused(view);
                        alpaka::ignore_unused(rowSeparator);

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
            //! Prints the content of the view to the given queue.
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
                    ALPAKA_NO_HOST_ACC_WARNING
                    template<
                        typename TPitch>
                    ALPAKA_FN_HOST_ACC
                    static auto create(
                        TPitch const & pitch)
                    -> idx::Idx<TPitch>
                    {
                        return mem::view::getPitchBytes<Tidx>(pitch);
                    }
                };
            }
            //-----------------------------------------------------------------------------
            //! \return The pitch vector.
            template<
                typename TPitch>
            auto getPitchBytesVec(
                TPitch const & pitch = TPitch())
            -> vec::Vec<dim::Dim<TPitch>, idx::Idx<TPitch>>
            {
                return
                    vec::createVecFromIndexedFn<
                        dim::Dim<TPitch>,
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
            -> vec::Vec<TDim, idx::Idx<TPitch>>
            {
                using IdxOffset = std::integral_constant<std::intmax_t, static_cast<std::intmax_t>(dim::Dim<TPitch>::value) - static_cast<std::intmax_t>(TDim::value)>;
                return
                    vec::createVecFromIndexedFnOffset<
                        TDim,
                        detail::CreatePitchBytes,
                        IdxOffset>(
                            pitch);
            }

            //-----------------------------------------------------------------------------
            //! \return A view to static device memory.
            template<
                typename TElem,
                typename TDev,
                typename TExtent>
            auto createStaticDevMemView(
                TElem * pMem,
                TDev const & dev,
                TExtent const & extent)
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
