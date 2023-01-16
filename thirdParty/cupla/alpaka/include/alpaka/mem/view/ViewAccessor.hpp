/* Copyright 2022 Bernhard Manfred Gruber

 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#pragma once

#include <alpaka/dim/Traits.hpp>
#include <alpaka/mem/view/Accessor.hpp>

#include <type_traits>

namespace alpaka::experimental
{
    namespace internal
    {
        template<typename T>
        ALPAKA_FN_HOST_ACC auto asBytePtr(T* p)
        {
            return reinterpret_cast<char*>(p);
        }

        template<typename T>
        struct WriteOnlyProxy
        {
            ALPAKA_FN_HOST_ACC WriteOnlyProxy(T& location) : loc(location)
            {
            }

            template<typename U>
            ALPAKA_FN_HOST_ACC auto operator=(U&& value) -> auto&
            {
                loc = std::forward<U>(value);
                return *this;
            }

        private:
            T& loc;
        };

        template<typename TElem, typename TAccessModes>
        struct AccessReturnTypeImpl;

        template<typename TElem>
        struct AccessReturnTypeImpl<TElem, ReadAccess>
        {
            using type = TElem;
        };

        template<typename TElem>
        struct AccessReturnTypeImpl<TElem, WriteAccess>
        {
            using type = WriteOnlyProxy<TElem>;
        };

        template<typename TElem>
        struct AccessReturnTypeImpl<TElem, ReadWriteAccess>
        {
            using type = TElem&;
        };

        template<typename TElem, typename THeadAccessMode, typename... TTailAccessModes>
        struct AccessReturnTypeImpl<TElem, std::tuple<THeadAccessMode, TTailAccessModes...>>
            : AccessReturnTypeImpl<TElem, THeadAccessMode>
        {
        };

        template<typename TElem, typename TAccessModes>
        using AccessReturnType = typename internal::AccessReturnTypeImpl<TElem, TAccessModes>::type;
    } // namespace internal

    //! 1D accessor to memory objects represented by a pointer.
    // We keep this specialization to not store the zero-dim pitch vector and provide one more operator[].
    template<typename TElem, typename TBufferIdx, typename TAccessModes>
    struct Accessor<TElem*, TElem, TBufferIdx, 1, TAccessModes>
    {
        using ReturnType = internal::AccessReturnType<TElem, TAccessModes>;

        ALPAKA_FN_HOST_ACC Accessor(
            TElem* p_,
            Vec<DimInt<0>, TBufferIdx> pitchesInBytes_,
            Vec<DimInt<1>, TBufferIdx> extents_)
            : p(p_)
            , extents(extents_)
        {
            (void) pitchesInBytes_;
        }

        template<typename TOtherAccessModes>
        ALPAKA_FN_HOST_ACC Accessor(Accessor<TElem*, TElem, TBufferIdx, 1, TOtherAccessModes> const& other)
            : p(other.p)
            , extents(other.extents)
        {
        }

        ALPAKA_FN_HOST_ACC auto operator[](Vec<DimInt<1>, TBufferIdx> i) const -> ReturnType
        {
            return (*this)(i[0]);
        }

        ALPAKA_FN_HOST_ACC auto operator[](TBufferIdx i) const -> ReturnType
        {
            return (*this)(i);
        }

        ALPAKA_FN_HOST_ACC auto operator()(TBufferIdx i) const -> ReturnType
        {
            return p[i];
        }

        TElem* p;
        Vec<DimInt<1>, TBufferIdx> extents;
    };

    //! Higher than 1D accessor to memory objects represented by a pointer.
    template<typename TElem, typename TBufferIdx, std::size_t TDim, typename TAccessModes>
    struct Accessor<TElem*, TElem, TBufferIdx, TDim, TAccessModes>
    {
        using ReturnType = internal::AccessReturnType<TElem, TAccessModes>;

        ALPAKA_FN_HOST_ACC Accessor(
            TElem* p_,
            Vec<DimInt<TDim - 1>, TBufferIdx> pitchesInBytes_,
            Vec<DimInt<TDim>, TBufferIdx> extents_)
            : p(p_)
            , pitchesInBytes(pitchesInBytes_)
            , extents(extents_)
        {
        }

        template<typename TOtherAccessModes>
        ALPAKA_FN_HOST_ACC Accessor(Accessor<TElem*, TElem, TBufferIdx, TDim, TOtherAccessModes> const& other)
            : p(other.p)
            , pitchesInBytes(other.pitchesInBytes)
            , extents(other.extents)
        {
        }

    private:
        template<std::size_t... TIs>
        [[nodiscard]] ALPAKA_FN_HOST_ACC auto subscript(Vec<DimInt<TDim>, TBufferIdx> index) const -> ReturnType
        {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-align"
#endif
            auto bp = internal::asBytePtr(p);
            for(std::size_t i = 0u; i < TDim; i++)
            {
                auto const pitch = i < TDim - 1 ? pitchesInBytes[i] : static_cast<TBufferIdx>(sizeof(TElem));
                bp += index[i] * pitch;
            }
            return *reinterpret_cast<TElem*>(bp);
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        }

    public:
        ALPAKA_FN_HOST_ACC auto operator[](Vec<DimInt<TDim>, TBufferIdx> i) const -> ReturnType
        {
            return subscript(i);
        }

        template<typename... Ts>
        ALPAKA_FN_HOST_ACC auto operator()(Ts... i) const -> ReturnType
        {
            static_assert(sizeof...(Ts) == TDim, "You need to specify TDim indices.");
            return subscript(Vec<DimInt<TDim>, TBufferIdx>{static_cast<TBufferIdx>(i)...});
        }

        TElem* p;
        Vec<DimInt<TDim - 1>, TBufferIdx> pitchesInBytes;
        Vec<DimInt<TDim>, TBufferIdx> extents;
    };

    namespace trait
    {
        namespace internal
        {
            template<typename T, typename SFINAE = void>
            struct IsView : std::false_type
            {
            };

            // TODO: replace this by a concept in C++20
            template<typename TView>
            struct IsView<
                TView,
                std::void_t<
                    Idx<TView>,
                    Dim<TView>,
                    decltype(getPtrNative(std::declval<TView>())),
                    decltype(getPitchBytes<0>(std::declval<TView>())),
                    decltype(getExtent<0>(std::declval<TView>()))>> : std::true_type
            {
            };

            template<typename... TAccessModes>
            struct BuildAccessModeList;

            template<typename TAccessMode>
            struct BuildAccessModeList<TAccessMode>
            {
                using type = TAccessMode;
            };

            template<typename TAccessMode1, typename TAccessMode2, typename... TAccessModes>
            struct BuildAccessModeList<TAccessMode1, TAccessMode2, TAccessModes...>
            {
                using type = std::tuple<TAccessMode1, TAccessMode2, TAccessModes...>;
            };

            template<
                typename... TAccessModes,
                typename TViewForwardRef,
                std::size_t... TPitchIs,
                std::size_t... TExtentIs>
            ALPAKA_FN_HOST auto buildViewAccessor(
                TViewForwardRef&& view,
                std::index_sequence<TPitchIs...>,
                std::index_sequence<TExtentIs...>)
            {
                using TView = std::decay_t<TViewForwardRef>;
                static_assert(IsView<TView>::value);
                using TBufferIdx = Idx<TView>;
                constexpr auto dim = Dim<TView>::value;
                using Elem = Elem<TView>;
                auto p = getPtrNative(view);
                static_assert(
                    std::is_same_v<decltype(p), Elem const*> || std::is_same_v<decltype(p), Elem*>,
                    "We assume that getPtrNative() returns a raw pointer to the view's elements");
                static_assert(
                    !std::is_same_v<
                        decltype(p),
                        Elem const*> || std::is_same_v<std::tuple<TAccessModes...>, std::tuple<ReadAccess>>,
                    "When getPtrNative() returns a const raw pointer, the access mode must be ReadAccess");
                using AccessModeList = typename BuildAccessModeList<TAccessModes...>::type;
                return Accessor<Elem*, Elem, TBufferIdx, dim, AccessModeList>{
                    const_cast<Elem*>(p), // strip constness, this is handled the the access modes
                    {getPitchBytes<TPitchIs + 1>(view)...},
                    {getExtent<TExtentIs>(view)...}};
            }
        } // namespace internal

        //! Builds an accessor from view like memory objects.
        template<typename TView>
        struct BuildAccessor<TView, std::enable_if_t<internal::IsView<TView>::value>>
        {
            template<typename... TAccessModes, typename TViewForwardRef>
            ALPAKA_FN_HOST static auto buildAccessor(TViewForwardRef&& view)
            {
                using Dim = Dim<std::decay_t<TView>>;
                return internal::buildViewAccessor<TAccessModes...>(
                    std::forward<TViewForwardRef>(view),
                    std::make_index_sequence<Dim::value - 1>{},
                    std::make_index_sequence<Dim::value>{});
            }
        };
    } // namespace trait
} // namespace alpaka::experimental
