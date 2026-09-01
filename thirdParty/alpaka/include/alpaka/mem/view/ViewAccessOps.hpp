/* Copyright 2025 Andrea Bocci, Bernhard Manfred Gruber, Jan Stephan, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"

#include <cstdint>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace alpaka
{
    class DevCpu;
} // namespace alpaka

namespace alpaka::internal
{

    template<typename TView>
    concept ViewType = requires {
        typename Idx<TView>;
        typename Dim<TView>;
        {
            getPtrNative(std::declval<TView>())
        };
        {
            getPitchesInBytes(std::declval<TView>())
        };
        {
            getExtents(std::declval<TView>())
        };
    };

    template<ViewType TView>
    struct BaseViewAccessor
    {
    private:
        using value_type = Elem<TView>;
        using pointer = value_type*;
        using const_pointer = value_type const*;
        using reference = value_type&;
        using const_reference = value_type const&;
        using Idx = alpaka::Idx<TView>;
        using Dim = alpaka::Dim<TView>;

    public:
        [[nodiscard]] ALPAKA_FN_HOST auto data() -> pointer
        {
            return getPtrNative(*static_cast<TView*>(this));
        }

        [[nodiscard]] ALPAKA_FN_HOST auto data() const -> const_pointer
        {
            return getPtrNative(*static_cast<TView const*>(this));
        }

        ALPAKA_FN_HOST auto begin() -> pointer requires(Dim::value == 1)
        {
            return data();
        }

        ALPAKA_FN_HOST auto begin() const -> const_pointer requires(Dim::value == 1)
        {
            return data();
        }

        ALPAKA_FN_HOST auto cbegin() const -> const_pointer requires(Dim::value == 1)
        {
            return data();
        }

        ALPAKA_FN_HOST auto end() -> pointer requires(Dim::value == 1)
        {
            return data() + getExtents(*static_cast<TView*>(this))[0];
        }

        ALPAKA_FN_HOST auto end() const -> const_pointer requires(Dim::value == 1)
        {
            return data() + getExtents(*static_cast<TView const*>(this))[0];
        }

        ALPAKA_FN_HOST auto cend() const -> const_pointer requires(Dim::value == 1)
        {
            return data() + getExtents(*static_cast<TView const*>(this))[0];
        }

        ALPAKA_FN_HOST auto rank() const -> Idx
        {
            return Dim::value;
        }

        ALPAKA_FN_HOST auto size() const -> Idx requires(Dim::value == 1)
        {
            return getExtents(*static_cast<TView const*>(this))[0];
        }

        ALPAKA_FN_HOST auto size() const -> Idx requires(Dim::value > 1)
        {
            return getExtents(*static_cast<TView const*>(this)).prod();
        }

        ALPAKA_FN_HOST auto extent(Idx dim) const -> Idx
        {
            return getExtents(*static_cast<TView const*>(this))[dim];
        }

        ALPAKA_FN_HOST auto extents() const -> Vec<Dim, Idx>;

#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif
        ALPAKA_FN_HOST operator std::span<value_type const>() const requires(Dim::value == 1)
        {
            return std::span<value_type const>{begin(), end()};
        }

        ALPAKA_FN_HOST operator std::span<value_type>() requires(Dim::value == 1)
        {
            return std::span<value_type>{begin(), end()};
        }
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    };

    template<ViewType TView>
    using DeviceViewAccessor = BaseViewAccessor<TView>;

    template<ViewType TView>
    struct HostViewAccessor : BaseViewAccessor<TView>
    {
    private:
        using value_type = Elem<TView>;
        using pointer = value_type*;
        using const_pointer = value_type const*;
        using reference = value_type&;
        using const_reference = value_type const&;
        using Idx = alpaka::Idx<TView>;
        using Dim = alpaka::Dim<TView>;

    public:
        ALPAKA_FN_HOST auto operator*() -> reference
        {
            static_assert(Dim::value == 0, "operator* is only valid for Buffers and Views of dimension 0");
            return *(this->data());
        }

        ALPAKA_FN_HOST auto operator*() const -> const_reference
        {
            static_assert(Dim::value == 0, "operator* is only valid for Buffers and Views of dimension 0");
            return *(this->data());
        }

        ALPAKA_FN_HOST auto operator->() -> pointer
        {
            static_assert(Dim::value == 0, "operator-> is only valid for Buffers and Views of dimension 0");
            return this->data();
        }

        ALPAKA_FN_HOST auto operator->() const -> const_pointer
        {
            static_assert(Dim::value == 0, "operator-> is only valid for Buffers and Views of dimension 0");
            return this->data();
        }

        ALPAKA_FN_HOST auto operator[](Idx i) -> reference
        {
            static_assert(Dim::value == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return this->data()[i];
        }

        ALPAKA_FN_HOST auto operator[](Idx i) const -> const_reference
        {
            static_assert(Dim::value == 1, "operator[i] is only valid for Buffers and Views of dimension 1");
            return this->data()[i];
        }

    private:
        template<typename TIdx>
        [[nodiscard]] ALPAKA_FN_HOST auto ptr_at([[maybe_unused]] Vec<Dim, TIdx> index) const -> const_pointer
        {
            static_assert(
                std::is_convertible_v<TIdx, Idx>,
                "the index type must be convertible to the index of the Buffer or View");

            auto ptr = reinterpret_cast<std::uintptr_t>(this->data());
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

    template<typename TDev>
    struct ViewAccessor
    {
        template<ViewType TView>
        using AccessorType = DeviceViewAccessor<TView>;
    };

    template<>
    struct ViewAccessor<alpaka::DevCpu>
    {
        template<ViewType TView>
        using AccessorType = HostViewAccessor<TView>;
    };

#ifdef ALPAKA_ACC_SYCL_ENABLED
    template<>
    struct ViewAccessor<alpaka::DevGenericSycl<alpaka::TagCpuSycl>>
    {
        template<ViewType TView>
        using AccessorType = HostViewAccessor<TView>;
    };
#endif

    template<typename TDev, ViewType TView>
    using ViewAccessorType = typename ViewAccessor<TDev>::template AccessorType<TView>;

} // namespace alpaka::internal
