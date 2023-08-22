/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dim/Traits.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"

#include <sstream>
#include <type_traits>

namespace alpaka::internal
{
    template<typename T, typename SFINAE = void>
    inline constexpr bool isView = false;

    // TODO(bgruber): replace this by a concept in C++20
    template<typename TView>
    inline constexpr bool isView<
        TView,
        std::void_t<
            Idx<TView>,
            Dim<TView>,
            decltype(getPtrNative(std::declval<TView>())),
            decltype(getPitchBytes<0>(std::declval<TView>())),
            decltype(getExtent<0>(std::declval<TView>()))>> = true;

    template<typename TView>
    struct ViewAccessOps
    {
        static_assert(isView<TView>);

    private:
        using value_type = Elem<TView>;
        using pointer = value_type*;
        using const_pointer = value_type const*;
        using reference = value_type&;
        using const_reference = value_type const&;
        using Idx = alpaka::Idx<TView>;

    public:
        ALPAKA_FN_HOST auto data() -> pointer
        {
            return getPtrNative(*static_cast<TView*>(this));
        }

        [[nodiscard]] ALPAKA_FN_HOST auto data() const -> const_pointer
        {
            return getPtrNative(*static_cast<TView const*>(this));
        }

        ALPAKA_FN_HOST auto operator*() -> reference
        {
            static_assert(Dim<TView>::value == 0, "operator* is only valid for Buffers and Views of dimension 0");
            return *data();
        }

        ALPAKA_FN_HOST auto operator*() const -> const_reference
        {
            static_assert(Dim<TView>::value == 0, "operator* is only valid for Buffers and Views of dimension 0");
            return *data();
        }

        ALPAKA_FN_HOST auto operator->() -> pointer
        {
            static_assert(Dim<TView>::value == 0, "operator-> is only valid for Buffers and Views of dimension 0");
            return data();
        }

        ALPAKA_FN_HOST auto operator->() const -> const_pointer
        {
            static_assert(Dim<TView>::value == 0, "operator-> is only valid for Buffers and Views of dimension 0");
            return data();
        }

        ALPAKA_FN_HOST auto operator[](Idx i) -> reference
        {
            static_assert(Dim<TView>::value == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return data()[i];
        }

        ALPAKA_FN_HOST auto operator[](Idx i) const -> const_reference
        {
            static_assert(Dim<TView>::value == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return data()[i];
        }

    private:
        template<std::size_t TDim, typename TIdx>
        [[nodiscard]] ALPAKA_FN_HOST auto ptr_at([[maybe_unused]] Vec<DimInt<TDim>, TIdx> index) const -> const_pointer
        {
            static_assert(
                Dim<TView>::value == TDim,
                "the index type must have the same dimensionality as the Buffer or View");
            static_assert(
                std::is_convertible_v<TIdx, Idx>,
                "the index type must be convertible to the index of the Buffer or View");

            auto ptr = reinterpret_cast<uintptr_t>(data());
            if constexpr(TDim > 0)
            {
                auto const pitchesInBytes = getPitchBytesVec(*static_cast<TView const*>(this));
                for(std::size_t i = 0u; i < TDim; i++)
                {
                    const Idx pitch = i + 1 < TDim ? pitchesInBytes[i + 1] : static_cast<Idx>(sizeof(value_type));
                    ptr += static_cast<uintptr_t>(index[i] * pitch);
                }
            }
            return reinterpret_cast<const_pointer>(ptr);
        }

    public:
        template<std::size_t TDim, typename TIdx>
        ALPAKA_FN_HOST auto operator[](Vec<DimInt<TDim>, TIdx> index) -> reference
        {
            return *const_cast<pointer>(ptr_at(index));
        }

        template<std::size_t TDim, typename TIdx>
        ALPAKA_FN_HOST auto operator[](Vec<DimInt<TDim>, TIdx> index) const -> const_reference
        {
            return *ptr_at(index);
        }

        template<std::size_t TDim, typename TIdx>
        ALPAKA_FN_HOST auto at(Vec<DimInt<TDim>, TIdx> index) -> reference
        {
            auto extent = getExtentVec(*static_cast<TView*>(this));
            if(!(index < extent).foldrAll(std::logical_and<bool>(), true))
            {
                std::stringstream msg;
                msg << "index " << index << " is outside of the Buffer or View extent " << extent;
                throw std::out_of_range(msg.str());
            }
            return *const_cast<pointer>(ptr_at(index));
        }

        template<std::size_t TDim, typename TIdx>
        [[nodiscard]] ALPAKA_FN_HOST auto at(Vec<DimInt<TDim>, TIdx> index) const -> const_reference
        {
            auto extent = getExtentVec(*static_cast<TView const*>(this));
            if(!(index < extent).foldrAll(std::logical_and<bool>(), true))
            {
                std::stringstream msg;
                msg << "index " << index << " is outside of the Buffer or View extent " << extent;
                throw std::out_of_range(msg.str());
            }
            return *ptr_at(index);
        }
    };
} // namespace alpaka::internal
