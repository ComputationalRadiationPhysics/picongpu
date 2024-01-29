/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/elem/Traits.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/meta/Fold.hpp"
#include "alpaka/meta/Integral.hpp"
#include "alpaka/offset/Traits.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/vec/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <array>
#include <cstddef>
#include <iosfwd>
#include <type_traits>
#include <vector>
#ifdef ALPAKA_USE_MDSPAN
#    include <experimental/mdspan>
#endif

namespace alpaka
{
    namespace detail
    {
        //! Calculate the pitches purely from the extents.
        template<typename TElem, typename TDim, typename TIdx>
        ALPAKA_FN_HOST_ACC inline constexpr auto calculatePitchesFromExtents(Vec<TDim, TIdx> const& extent)
        {
            Vec<TDim, TIdx> pitchBytes{};
            constexpr auto dim = TIdx{TDim::value};
            if constexpr(dim > 0)
                pitchBytes.back() = static_cast<TIdx>(sizeof(TElem));
            if constexpr(dim > 1)
                for(TIdx i = TDim::value - 1; i > 0; i--)
                    pitchBytes[i - 1] = extent[i] * pitchBytes[i];
            return pitchBytes;
        }
    } // namespace detail

    //! The view traits.
    namespace trait
    {
        //! The native pointer get trait.
        template<typename TView, typename TSfinae = void>
        struct GetPtrNative;

        //! The pointer on device get trait.
        template<typename TView, typename TDev, typename TSfinae = void>
        struct GetPtrDev;

        //! The pitch in bytes.
        //! This is the distance in bytes in the linear memory between two consecutive elements in the next higher
        //! dimension (TIdx-1).
        //!
        //! The default implementation uses the extent to calculate the pitch.
        template<typename TIdx, typename TView, typename TSfinae = void>
        struct [[deprecated("Use GetPitchesInBytes instead")]] GetPitchBytes
        {
            using ViewIdx = Idx<TView>;

            ALPAKA_FN_HOST static auto getPitchBytes(TView const& view) -> ViewIdx
            {
                return getPitchBytesDefault(view);
            }

        private:
            static auto getPitchBytesDefault(TView const& view) -> ViewIdx
            {
                constexpr auto idx = TIdx::value;
                constexpr auto viewDim = Dim<TView>::value;
                if constexpr(idx < viewDim - 1)
                {
#if BOOST_COMP_CLANG || BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
                    return getExtents(view)[idx] * GetPitchBytes<DimInt<idx + 1>, TView>::getPitchBytes(view);
#if BOOST_COMP_CLANG || BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
                }
                else if constexpr(idx == viewDim - 1)
                    return getExtents(view)[viewDim - 1] * static_cast<ViewIdx>(sizeof(Elem<TView>));
                else
                    return static_cast<ViewIdx>(sizeof(Elem<TView>));
                ALPAKA_UNREACHABLE({});
            }
        };

        //! Customization point for \ref getPitchesInBytes.
        //! The default implementation uses the extent to calculate the pitches.
        template<typename TView, typename TSfinae = void>
        struct GetPitchesInBytes
        {
            ALPAKA_FN_HOST_ACC constexpr auto operator()(TView const& view) const
            {
                return alpaka::detail::calculatePitchesFromExtents<Elem<TView>>(getExtents(view));
            }
        };

        //! The memory set task trait.
        //!
        //! Fills the view with data.
        template<typename TDim, typename TDev, typename TSfinae = void>
        struct CreateTaskMemset;

        //! The memory copy task trait.
        //!
        //! Copies memory from one view into another view possibly on a different device.
        template<typename TDim, typename TDevDst, typename TDevSrc, typename TSfinae = void>
        struct CreateTaskMemcpy;

        //! The static device memory view creation trait.
        template<typename TDev, typename TSfinae = void>
        struct CreateStaticDevMemView;

        //! The device memory view creation trait.
        template<typename TDev, typename TSfinae = void>
        struct CreateViewPlainPtr;

        //! The sub view creation trait.
        template<typename TDev, typename TSfinae = void>
        struct CreateSubView;
    } // namespace trait

    //! Gets the native pointer of the memory view.
    //!
    //! \param view The memory view.
    //! \return The native pointer.
    template<typename TView>
    ALPAKA_FN_HOST auto getPtrNative(TView const& view) -> Elem<TView> const*
    {
        return trait::GetPtrNative<TView>::getPtrNative(view);
    }

    //! Gets the native pointer of the memory view.
    //!
    //! \param view The memory view.
    //! \return The native pointer.
    template<typename TView>
    ALPAKA_FN_HOST auto getPtrNative(TView& view) -> Elem<TView>*
    {
        return trait::GetPtrNative<TView>::getPtrNative(view);
    }

    //! Gets the pointer to the view on the given device.
    //!
    //! \param view The memory view.
    //! \param dev The device.
    //! \return The pointer on the device.
    template<typename TView, typename TDev>
    ALPAKA_FN_HOST auto getPtrDev(TView const& view, TDev const& dev) -> Elem<TView> const*
    {
        return trait::GetPtrDev<TView, TDev>::getPtrDev(view, dev);
    }

    //! Gets the pointer to the view on the given device.
    //!
    //! \param view The memory view.
    //! \param dev The device.
    //! \return The pointer on the device.
    template<typename TView, typename TDev>
    ALPAKA_FN_HOST auto getPtrDev(TView& view, TDev const& dev) -> Elem<TView>*
    {
        return trait::GetPtrDev<TView, TDev>::getPtrDev(view, dev);
    }

    //! \return The pitch in bytes. This is the distance in bytes between two consecutive elements in the given
    //! dimension.
    template<std::size_t Tidx, typename TView>
    [[deprecated("Use getPitchesInBytes instead")]] ALPAKA_FN_HOST auto getPitchBytes(TView const& view) -> Idx<TView>
    {
#if BOOST_COMP_CLANG || BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return trait::GetPitchBytes<DimInt<Tidx>, TView>::getPitchBytes(view);
#if BOOST_COMP_CLANG || BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
    }

    //! \return The pitches in bytes as an alpaka::Vec. This is the distance in bytes between two consecutive elements
    //! in the given dimension.
    //! E.g. for a 3D view without padding, the 0-dim pitch is the distance in bytes to jump from one element to the
    //! next within the same row, the 1-dim pitch (aka. the row pitch) is the distance in bytes to jump from one
    //! element to the neighboring element on the next row. The 2-dim pitch (aka. the slice pitch) is the distance in
    //! bytes to jump from one element to the neighboring element on the next slice.
    //! E.g. a 3D view of floats without padding and the extents {42, 10, 2}, would have a pitch vector of {80, 8, 4}.
    template<typename TView>
    ALPAKA_FN_HOST auto getPitchesInBytes(TView const& view) -> Vec<Dim<TView>, Idx<TView>>
    {
        return trait::GetPitchesInBytes<TView>{}(view);
    }

    //! Create a memory set task.
    //!
    //! \param view The memory view to fill.
    //! \param byte Value to set for each element of the specified view.
    //! \param extent The extent of the view to fill.
    template<typename TExtent, typename TViewFwd>
    ALPAKA_FN_HOST auto createTaskMemset(TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent)
    {
        using TView = std::remove_reference_t<TViewFwd>;
        static_assert(!std::is_const_v<TView>, "The view must not be const!");
        static_assert(
            Dim<TView>::value == Dim<TExtent>::value,
            "The view and the extent are required to have the same dimensionality!");
        static_assert(
            meta::IsIntegralSuperset<Idx<TView>, Idx<TExtent>>::value,
            "The view and the extent must have compatible index types!");

        return trait::CreateTaskMemset<Dim<TView>, Dev<TView>>::createTaskMemset(
            std::forward<TViewFwd>(view),
            byte,
            extent);
    }

    //! Sets the bytes of the memory of view, described by extent, to the given value.
    //!
    //! \param queue The queue to enqueue the view fill task into.
    //! \param[in,out] view The memory view to fill. May be a temporary object.
    //! \param byte Value to set for each element of the specified view.
    //! \param extent The extent of the view to fill.
    template<typename TExtent, typename TViewFwd, typename TQueue>
    ALPAKA_FN_HOST auto memset(TQueue& queue, TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent) -> void
    {
        enqueue(queue, createTaskMemset(std::forward<TViewFwd>(view), byte, extent));
    }

    //! Sets each byte of the memory of the entire view to the given value.
    //!
    //! \param queue The queue to enqueue the view fill task into.
    //! \param[in,out] view The memory view to fill. May be a temporary object.
    //! \param byte Value to set for each element of the specified view.
    template<typename TViewFwd, typename TQueue>
    ALPAKA_FN_HOST auto memset(TQueue& queue, TViewFwd&& view, std::uint8_t const& byte) -> void
    {
        enqueue(queue, createTaskMemset(std::forward<TViewFwd>(view), byte, getExtents(view)));
    }

    //! Creates a memory copy task.
    //!
    //! \param viewDst The destination memory view.
    //! \param viewSrc The source memory view.
    //! \param extent The extent of the view to copy.
    template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
    ALPAKA_FN_HOST auto createTaskMemcpy(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
    {
        using TViewDst = std::remove_reference_t<TViewDstFwd>;
        using SrcElem = Elem<TViewSrc>;
        using DstElem = Elem<TViewDst>;
        using ExtentIdx = Idx<TExtent>;
        using DstIdx = Idx<TViewDst>;
        using SrcIdx = Idx<TViewSrc>;

        static_assert(!std::is_const_v<TViewDst>, "The destination view must not be const!");
        static_assert(!std::is_const_v<DstElem>, "The destination view's element type must not be const!");
        static_assert(
            Dim<TViewDst>::value == Dim<TViewSrc>::value,
            "The source and the destination view must have the same dimensionality!");
        static_assert(
            Dim<TViewDst>::value == Dim<TExtent>::value,
            "The destination view and the extent must have the same dimensionality!");
        static_assert(
            std::is_same_v<DstElem, std::remove_const_t<SrcElem>>,
            "The source and destination view must have the same element type!");
        static_assert(
            meta::IsIntegralSuperset<DstIdx, ExtentIdx>::value,
            "The destination view and the extent are required to have compatible index types!");
        static_assert(
            meta::IsIntegralSuperset<SrcIdx, ExtentIdx>::value,
            "The source view and the extent are required to have compatible index types!");

        return trait::CreateTaskMemcpy<Dim<TViewDst>, Dev<TViewDst>, Dev<TViewSrc>>::createTaskMemcpy(
            std::forward<TViewDstFwd>(viewDst),
            viewSrc,
            extent);
    }

    //! Copies memory from a part of viewSrc to viewDst, described by extent. Possibly copies between different memory
    //! spaces.
    //!
    //! \param queue The queue to enqueue the view copy task into.
    //! \param[in,out] viewDst The destination memory view. May be a temporary object.
    //! \param viewSrc The source memory view. May be a temporary object.
    //! \param extent The extent of the view to copy.
    template<typename TExtent, typename TViewSrc, typename TViewDstFwd, typename TQueue>
    ALPAKA_FN_HOST auto memcpy(TQueue& queue, TViewDstFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
        -> void
    {
        enqueue(queue, createTaskMemcpy(std::forward<TViewDstFwd>(viewDst), viewSrc, extent));
    }

    //! Copies the entire memory of viewSrc to viewDst. Possibly copies between different memory
    //! spaces.
    //!
    //! \param queue The queue to enqueue the view copy task into.
    //! \param[in,out] viewDst The destination memory view. May be a temporary object.
    //! \param viewSrc The source memory view. May be a temporary object.
    template<typename TViewSrc, typename TViewDstFwd, typename TQueue>
    ALPAKA_FN_HOST auto memcpy(TQueue& queue, TViewDstFwd&& viewDst, TViewSrc const& viewSrc) -> void
    {
        enqueue(queue, createTaskMemcpy(std::forward<TViewDstFwd>(viewDst), viewSrc, getExtents(viewSrc)));
    }

    namespace detail
    {
        template<typename TDim, typename TView>
        struct Print
        {
            ALPAKA_FN_HOST static auto print(
                TView const& view,
                Elem<TView> const* const ptr,
                Vec<Dim<TView>, Idx<TView>> const& extent,
                std::ostream& os,
                std::string const& elementSeparator,
                std::string const& rowSeparator,
                std::string const& rowPrefix,
                std::string const& rowSuffix) -> void
            {
                os << rowPrefix;

                auto const pitch = getPitchesInBytes(view)[TDim::value + 1];
                auto const lastIdx(extent[TDim::value] - 1u);
                for(auto i(decltype(lastIdx)(0)); i <= lastIdx; ++i)
                {
                    Print<DimInt<TDim::value + 1u>, TView>::print(
                        view,
                        reinterpret_cast<Elem<TView> const*>(reinterpret_cast<std::uint8_t const*>(ptr) + i * pitch),
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

        template<typename TView>
        struct Print<DimInt<Dim<TView>::value - 1u>, TView>
        {
            ALPAKA_FN_HOST static auto print(
                TView const& /* view */,
                Elem<TView> const* const ptr,
                Vec<Dim<TView>, Idx<TView>> const& extent,
                std::ostream& os,
                std::string const& elementSeparator,
                std::string const& /* rowSeparator */,
                std::string const& rowPrefix,
                std::string const& rowSuffix) -> void
            {
                os << rowPrefix;

                auto const lastIdx(extent[Dim<TView>::value - 1u] - 1u);
                for(auto i(decltype(lastIdx)(0)); i <= lastIdx; ++i)
                {
                    // Add the current element.
                    os << *(ptr + i);

                    // While we are not at the end of a line, add the element separator.
                    if(i != lastIdx)
                    {
                        os << elementSeparator;
                    }
                }

                os << rowSuffix;
            }
        };
    } // namespace detail

    //! Prints the content of the view to the given queue.
    // \TODO: Add precision flag.
    // \TODO: Add column alignment flag.
    template<typename TView>
    ALPAKA_FN_HOST auto print(
        TView const& view,
        std::ostream& os,
        std::string const& elementSeparator = ", ",
        std::string const& rowSeparator = "\n",
        std::string const& rowPrefix = "[",
        std::string const& rowSuffix = "]") -> void
    {
        detail::Print<DimInt<0u>, TView>::print(
            view,
            getPtrNative(view),
            getExtents(view),
            os,
            elementSeparator,
            rowSeparator,
            rowPrefix,
            rowSuffix);
    }

    //! \return The pitch vector.
    template<typename TView>
    [[deprecated("Use getPitchesInBytes instead")]] auto getPitchBytesVec(TView const& view)
        -> Vec<Dim<TView>, Idx<TView>>
    {
        return getPitchesInBytes(view);
    }

    //! \return The pitch but only the last N elements.
    template<typename TDim, typename TView>
    ALPAKA_FN_HOST auto getPitchBytesVecEnd(TView const& view = TView()) -> Vec<TDim, Idx<TView>>
    {
        return subVecEnd<TDim>(getPitchesInBytes(view));
    }

    //! \return A view to static device memory.
    template<typename TElem, typename TDev, typename TExtent>
    auto createStaticDevMemView(TElem* pMem, TDev const& dev, TExtent const& extent)
    {
        return trait::CreateStaticDevMemView<TDev>::createStaticDevMemView(pMem, dev, extent);
    }

    //! Creates a view to a device pointer
    //!
    //! \param dev Device from where pMem can be accessed.
    //! \param pMem Pointer to memory. The pointer must be accessible from the given device.
    //! \param extent Number of elements represented by the pMem.
    //!               Using a multi dimensional extent will result in a multi dimension view to the memory represented
    //!               by pMem.
    //! \return A view to device memory.
    template<typename TDev, typename TElem, typename TExtent>
    auto createView(TDev const& dev, TElem* pMem, TExtent const& extent)
    {
        using Dim = alpaka::Dim<TExtent>;
        using Idx = alpaka::Idx<TExtent>;
        auto const extentVec = Vec<Dim, Idx>(extent);
        return trait::CreateViewPlainPtr<TDev>::createViewPlainPtr(
            dev,
            pMem,
            extentVec,
            detail::calculatePitchesFromExtents<TElem>(extentVec));
    }

    //! Creates a view to a device pointer
    //!
    //! \param dev Device from where pMem can be accessed.
    //! \param pMem Pointer to memory. The pointer must be accessible from the given device.
    //! \param extent Number of elements represented by the pMem.
    //!               Using a multi dimensional extent will result in a multi dimension view to the memory represented
    //!               by pMem.
    //! \param pitch Pitch in bytes for each dimension. Dimensionality must be equal to extent.
    //! \return A view to device memory.
    template<typename TDev, typename TElem, typename TExtent, typename TPitch>
    auto createView(TDev const& dev, TElem* pMem, TExtent const& extent, TPitch pitch)
    {
        return trait::CreateViewPlainPtr<TDev>::createViewPlainPtr(dev, pMem, extent, pitch);
    }

    //! Creates a view to a contiguous container of device-accessible memory.
    //!
    //! \param dev Device from which the container can be accessed.
    //! \param con Contiguous container. The container must provide a `data()` method. The data held by the container
    //!            must be accessible from the given device. The `GetExtent` trait must be defined for the container.
    //! \return A view to device memory.
    template<typename TDev, typename TContainer>
    auto createView(TDev const& dev, TContainer& con)
    {
        return createView(dev, std::data(con), getExtents(con));
    }

    //! Creates a view to a contiguous container of device-accessible memory.
    //!
    //! \param dev Device from which the container can be accessed.
    //! \param con Contiguous container. The container must provide a `data()` method. The data held by the container
    //!            must be accessible from the given device. The `GetExtent` trait must be defined for the container.
    //! \param extent Number of elements held by the container. Using a multi-dimensional extent will result in a
    //!               multi-dimensional view to the memory represented by the container.
    //! \return A view to device memory.
    template<typename TDev, typename TContainer, typename TExtent>
    auto createView(TDev const& dev, TContainer& con, TExtent const& extent)
    {
        return createView(dev, std::data(con), extent);
    }

    //! Creates a sub view to an existing view.
    //!
    //! \param view The view this view is a sub-view of.
    //! \param extent Number of elements the resulting view holds.
    //! \param offset Number of elements skipped in view for the new origin of the resulting view.
    //! \return A sub view to a existing view.
    template<typename TView, typename TExtent, typename TOffsets>
    auto createSubView(TView& view, TExtent const& extent, TOffsets const& offset = TExtent())
    {
        return trait::CreateSubView<typename trait::DevType<TView>::type>::createSubView(view, extent, offset);
    }

#ifdef ALPAKA_USE_MDSPAN
    namespace experimental
    {
        // import mdspan into alpaka::experimental namespace. see: https://eel.is/c++draft/mdspan.syn
        using std::experimental::default_accessor;
        using std::experimental::dextents;
        using std::experimental::extents;
        using std::experimental::layout_left;
        using std::experimental::layout_right;
        using std::experimental::layout_stride;
        using std::experimental::mdspan;
        // import submdspan as well, which is not standardized yet
        using std::experimental::full_extent;
        using std::experimental::submdspan;

        namespace traits
        {
            namespace detail
            {
                template<typename ElementType>
                struct ByteIndexedAccessor
                {
                    using offset_policy = ByteIndexedAccessor;
                    using element_type = ElementType;
                    using reference = ElementType&;

                    using data_handle_type
                        = std::conditional_t<std::is_const_v<ElementType>, std::byte const*, std::byte*>;

                    constexpr ByteIndexedAccessor() noexcept = default;

                    ALPAKA_FN_HOST_ACC constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept
                    {
                        return p + i;
                    }

                    ALPAKA_FN_HOST_ACC constexpr reference access(data_handle_type p, size_t i) const noexcept
                    {
                        assert(i % alignof(ElementType) == 0);
#    if BOOST_COMP_GNUC
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wcast-align"
#    endif
                        return *reinterpret_cast<ElementType*>(p + i);
#    if BOOST_COMP_GNUC
#        pragma GCC diagnostic pop
#    endif
                    }
                };

                template<typename TView, std::size_t... Is>
                ALPAKA_FN_HOST auto makeExtents(TView const& view, std::index_sequence<Is...>)
                {
                    auto const ex = getExtents(view);
                    return std::experimental::dextents<Idx<TView>, Dim<TView>::value>{ex[Is]...};
                }
            } // namespace detail

            //! Customization point for getting an mdspan from a view.
            template<typename TView, typename TSfinae = void>
            struct GetMdSpan
            {
                ALPAKA_FN_HOST static auto getMdSpan(TView& view)
                {
                    constexpr auto dim = Dim<TView>::value;
                    using Element = Elem<TView>;
                    auto extents = detail::makeExtents(view, std::make_index_sequence<dim>{});
                    auto* ptr = reinterpret_cast<std::byte*>(getPtrNative(view));
                    auto const strides = toArray(getPitchesInBytes(view));
                    layout_stride::mapping<decltype(extents)> m{extents, strides};
                    return mdspan<Element, decltype(extents), layout_stride, detail::ByteIndexedAccessor<Element>>{
                        ptr,
                        m};
                }

                ALPAKA_FN_HOST static auto getMdSpanTransposed(TView& view)
                {
                    constexpr auto dim = Dim<TView>::value;
                    using Element = Elem<TView>;
                    auto extents = detail::makeExtents(view, std::make_index_sequence<dim>{});
                    auto* ptr = reinterpret_cast<std::byte*>(getPtrNative(view));
                    auto strides = toArray(getPitchesInBytes(view));
                    std::reverse(begin(strides), end(strides));
                    layout_stride::mapping<decltype(extents)> m{extents, strides};
                    return mdspan<Element, decltype(extents), layout_stride, detail::ByteIndexedAccessor<Element>>{
                        ptr,
                        m};
                }
            };
        } // namespace traits

        //! Gets a std::mdspan from the given view. The memory layout is determined by the pitches of the view.
        template<typename TView>
        ALPAKA_FN_HOST auto getMdSpan(TView& view)
        {
            return traits::GetMdSpan<TView>::getMdSpan(view);
        }

        //! Gets a std::mdspan from the given view. The memory layout is determined by the reversed pitches of the
        //! view. This effectively also reverses the extents of the view. In order words, if you create a transposed
        //! mdspan on a 10x5 element view, the mdspan will have an iteration space of 5x10.
        template<typename TView>
        ALPAKA_FN_HOST auto getMdSpanTransposed(TView& view)
        {
            return traits::GetMdSpan<TView>::getMdSpanTransposed(view);
        }

        template<typename TElem, typename TIdx, typename TDim>
        using MdSpan = alpaka::experimental::mdspan<
            TElem,
            alpaka::experimental::dextents<TIdx, TDim::value>,
            alpaka::experimental::layout_stride,
            alpaka::experimental::traits::detail::ByteIndexedAccessor<TElem>>;
    } // namespace experimental
#endif
} // namespace alpaka
