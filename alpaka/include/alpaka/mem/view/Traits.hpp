/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unreachable.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/elem/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/meta/Fold.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include <array>
#include <iosfwd>
#include <type_traits>
#include <vector>

namespace alpaka
{
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
        struct GetPitchBytes
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
                    return getExtent<idx>(view) * GetPitchBytes<DimInt<idx + 1>, TView>::getPitchBytes(view);
                else if constexpr(idx == viewDim - 1)
                    return getExtent<viewDim - 1>(view) * static_cast<ViewIdx>(sizeof(Elem<TView>));
                else
                    return static_cast<ViewIdx>(sizeof(Elem<TView>));
                ALPAKA_UNREACHABLE({});
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
    ALPAKA_NO_HOST_ACC_WARNING
    template<std::size_t Tidx, typename TView>
    ALPAKA_FN_HOST_ACC auto getPitchBytes(TView const& view) -> Idx<TView>
    {
        return trait::GetPitchBytes<DimInt<Tidx>, TView>::getPitchBytes(view);
    }

    //! Create a memory set task.
    //!
    //! \param view The memory view to fill.
    //! \param byte Value to set for each element of the specified view.
    //! \param extent The extent of the view to fill.
    template<typename TExtent, typename TView>
    ALPAKA_FN_HOST auto createTaskMemset(TView& view, std::uint8_t const& byte, TExtent const& extent)
    {
        static_assert(
            Dim<TView>::value == Dim<TExtent>::value,
            "The view and the extent are required to have the same dimensionality!");

        return trait::CreateTaskMemset<Dim<TView>, Dev<TView>>::createTaskMemset(view, byte, extent);
    }

    //! Sets the memory to the given value.
    //!
    //! \param queue The queue to enqueue the view fill task into.
    //! \param view The memory view to fill.
    //! \param byte Value to set for each element of the specified view.
    //! \param extent The extent of the view to fill.
    template<typename TExtent, typename TView, typename TQueue>
    ALPAKA_FN_HOST auto memset(TQueue& queue, TView& view, std::uint8_t const& byte, TExtent const& extent) -> void
    {
        enqueue(queue, createTaskMemset(view, byte, extent));
    }

    //! Sets the whole view to the given value.
    //!
    //! \param queue The queue to enqueue the view fill task into.
    //! \param view The memory view to fill.
    //! \param byte Value to set for each element of the specified view.
    template<typename TView, typename TQueue>
    ALPAKA_FN_HOST auto memset(TQueue& queue, TView& view, std::uint8_t const& byte) -> void
    {
        enqueue(queue, createTaskMemset(view, byte, getExtentVec(view)));
    }

    //! Creates a memory copy task.
    //!
    //! \param viewDst The destination memory view.
    //! \param viewSrc The source memory view.
    //! \param extent The extent of the view to copy.
    template<typename TExtent, typename TViewSrc, typename TViewDst>
    ALPAKA_FN_HOST auto createTaskMemcpy(TViewDst& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
    {
        static_assert(
            Dim<TViewDst>::value == Dim<TViewSrc>::value,
            "The source and the destination view are required to have the same dimensionality!");
        static_assert(
            Dim<TViewDst>::value == Dim<TExtent>::value,
            "The destination view and the extent are required to have the same dimensionality!");
        static_assert(
            std::is_same_v<Elem<TViewDst>, std::remove_const_t<Elem<TViewSrc>>>,
            "The source and the destination view are required to have the same element type!");

        return trait::CreateTaskMemcpy<Dim<TViewDst>, Dev<TViewDst>, Dev<TViewSrc>>::createTaskMemcpy(
            viewDst,
            viewSrc,
            extent);
    }

    //! Copies memory possibly between different memory spaces.
    //!
    //! \param queue The queue to enqueue the view copy task into.
    //! \param viewDst The destination memory view.
    //! \param viewSrc The source memory view.
    //! \param extent The extent of the view to copy.
    template<typename TExtent, typename TViewSrc, typename TViewDst, typename TQueue>
    ALPAKA_FN_HOST auto memcpy(TQueue& queue, TViewDst& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
        -> void
    {
        enqueue(queue, createTaskMemcpy(viewDst, viewSrc, extent));
    }

    //! Copies the whole view possibly between different memory spaces.
    //!
    //! \param queue The queue to enqueue the view copy task into.
    //! \param viewDst The destination memory view.
    //! \param viewSrc The source memory view.
    template<typename TViewSrc, typename TViewDst, typename TQueue>
    ALPAKA_FN_HOST auto memcpy(TQueue& queue, TViewDst& viewDst, TViewSrc const& viewSrc) -> void
    {
        enqueue(queue, createTaskMemcpy(viewDst, viewSrc, getExtentVec(viewSrc)));
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

                auto const pitch(getPitchBytes<TDim::value + 1u>(view));
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
            getExtentVec(view),
            os,
            elementSeparator,
            rowSeparator,
            rowPrefix,
            rowSuffix);
    }

    namespace detail
    {
        //! A class with a create method that returns the pitch for each index.
        template<std::size_t Tidx>
        struct CreatePitchBytes
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TView>
            ALPAKA_FN_HOST_ACC static auto create(TView const& view) -> Idx<TView>
            {
                return getPitchBytes<Tidx>(view);
            }
        };

        //! Calculate the pitches purely from the extents.
        template<typename TElem, typename TDim, typename TIdx>
        ALPAKA_FN_HOST inline auto calculatePitchesFromExtents(Vec<TDim, TIdx> const& extent)
        {
            Vec<TDim, TIdx> pitchBytes(Vec<TDim, TIdx>::all(0));
            if constexpr(TDim::value > 0)
            {
                pitchBytes[TDim::value - 1u] = extent[TDim::value - 1u] * static_cast<TIdx>(sizeof(TElem));
                for(TIdx i = TDim::value - 1u; i > static_cast<TIdx>(0u); --i)
                {
                    pitchBytes[i - 1] = extent[i - 1] * pitchBytes[i];
                }
            }
            return pitchBytes;
        }
    } // namespace detail
    //! \return The pitch vector.
    template<typename TView>
    auto getPitchBytesVec(TView const& view = TView()) -> Vec<Dim<TView>, Idx<TView>>
    {
        return createVecFromIndexedFn<Dim<TView>, detail::CreatePitchBytes>(view);
    }

    //! \return The pitch but only the last N elements.
    template<typename TDim, typename TView>
    ALPAKA_FN_HOST auto getPitchBytesVecEnd(TView const& view = TView()) -> Vec<TDim, Idx<TView>>
    {
        using IdxOffset = std::integral_constant<
            std::intmax_t,
            static_cast<std::intmax_t>(Dim<TView>::value) - static_cast<std::intmax_t>(TDim::value)>;
        return createVecFromIndexedFnOffset<TDim, detail::CreatePitchBytes, IdxOffset>(view);
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
    auto createView(TDev const& dev, TElem* pMem, TExtent const& extent, TPitch const& pitch)
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
        return createView(dev, std::data(con), getExtent(con));
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

} // namespace alpaka
