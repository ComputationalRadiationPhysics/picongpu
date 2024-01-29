/* Copyright 2023 Andrea Bocci, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dim/Traits.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

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
            decltype(getPitchesInBytes(std::declval<TView>())),
            decltype(getExtents(std::declval<TView>()))>>
        = true;

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
        using Dim = alpaka::Dim<TView>;

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
            static_assert(Dim::value == 0, "operator* is only valid for Buffers and Views of dimension 0");
            return *data();
        }

        ALPAKA_FN_HOST auto operator*() const -> const_reference
        {
            static_assert(Dim::value == 0, "operator* is only valid for Buffers and Views of dimension 0");
            return *data();
        }

        ALPAKA_FN_HOST auto operator->() -> pointer
        {
            static_assert(Dim::value == 0, "operator-> is only valid for Buffers and Views of dimension 0");
            return data();
        }

        ALPAKA_FN_HOST auto operator->() const -> const_pointer
        {
            static_assert(Dim::value == 0, "operator-> is only valid for Buffers and Views of dimension 0");
            return data();
        }

        ALPAKA_FN_HOST auto operator[](Idx i) -> reference
        {
            static_assert(Dim::value == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return data()[i];
        }

        ALPAKA_FN_HOST auto operator[](Idx i) const -> const_reference
        {
            static_assert(Dim::value == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return data()[i];
        }

    private:
        template<typename TIdx>
        [[nodiscard]] ALPAKA_FN_HOST auto ptr_at([[maybe_unused]] Vec<Dim, TIdx> index) const -> const_pointer
        {
            static_assert(
                std::is_convertible_v<TIdx, Idx>,
                "the index type must be convertible to the index of the Buffer or View");

            auto ptr = reinterpret_cast<std::uintptr_t>(data());
            if constexpr(Dim::value > 0)
            {
                ptr += static_cast<std::uintptr_t>(
                    (getPitchesInBytes(*static_cast<TView const*>(this)) * castVec<Idx>(index)).sum());
            }
            return reinterpret_cast<const_pointer>(ptr);
        }

    public:
        template<typename TIdx>
        ALPAKA_FN_HOST auto operator[](Vec<Dim, TIdx> index) -> reference
        {
            return *const_cast<pointer>(ptr_at(index));
        }

        template<typename TIdx>
        ALPAKA_FN_HOST auto operator[](Vec<Dim, TIdx> index) const -> const_reference
        {
            return *ptr_at(index);
        }

        template<typename TIdx>
        ALPAKA_FN_HOST auto at(Vec<Dim, TIdx> index) -> reference
        {
            auto extent = getExtents(*static_cast<TView*>(this));
            if(!(index < extent).all())
            {
                std::stringstream msg;
                msg << "index " << index << " is outside of the Buffer or View extent " << extent;
                throw std::out_of_range(msg.str());
            }
            return *const_cast<pointer>(ptr_at(index));
        }

        template<typename TIdx>
        [[nodiscard]] ALPAKA_FN_HOST auto at(Vec<Dim, TIdx> index) const -> const_reference
        {
            auto extent = getExtents(*static_cast<TView const*>(this));
            if(!(index < extent).all())
            {
                std::stringstream msg;
                msg << "index " << index << " is outside of the Buffer or View extent " << extent;
                throw std::out_of_range(msg.str());
            }
            return *ptr_at(index);
        }
    };
} // namespace alpaka::internal
