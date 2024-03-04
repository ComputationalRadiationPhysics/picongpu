/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/alpaka.hpp"

#include <type_traits>

namespace alpaka::test
{
    namespace trait
    {
        // \tparam T Type to conditionally make const.
        // \tparam TSource Type to mimic the constness of.
        template<typename T, typename TSource>
        using MimicConst = std::conditional_t<std::is_const_v<TSource>, std::add_const_t<T>, std::remove_const_t<T>>;

        template<typename TView, typename TSfinae = void>
        class IteratorView
        {
            using TViewDecayed = std::decay_t<TView>;
            using Dim = alpaka::Dim<TViewDecayed>;
            using Idx = alpaka::Idx<TViewDecayed>;
            using Elem = MimicConst<alpaka::Elem<TViewDecayed>, TView>;

        public:
            ALPAKA_FN_HOST IteratorView(TView& view, Idx const idx)
                : m_nativePtr(getPtrNative(view))
                , m_currentIdx(idx)
                , m_extents(getExtents(view))
                , m_pitchBytes(getPitchesInBytes(view))
            {
            }

            ALPAKA_FN_HOST explicit IteratorView(TView& view) : IteratorView(view, 0)
            {
            }

            ALPAKA_FN_HOST_ACC auto operator++() -> IteratorView&
            {
                ++m_currentIdx;
                return *this;
            }

            ALPAKA_FN_HOST_ACC auto operator--() -> IteratorView&
            {
                --m_currentIdx;
                return *this;
            }

            ALPAKA_FN_HOST_ACC auto operator++(int) -> IteratorView
            {
                IteratorView iterCopy = *this;
                m_currentIdx++;
                return iterCopy;
            }

            ALPAKA_FN_HOST_ACC auto operator--(int) -> IteratorView
            {
                IteratorView iterCopy = *this;
                m_currentIdx--;
                return iterCopy;
            }

            template<typename TIter>
            ALPAKA_FN_HOST_ACC auto operator==(TIter& other) const -> bool
            {
                return m_currentIdx == other.m_currentIdx;
            }

            template<typename TIter>
            ALPAKA_FN_HOST_ACC auto operator!=(TIter& other) const -> bool
            {
                return m_currentIdx != other.m_currentIdx;
            }

            ALPAKA_FN_HOST_ACC auto operator*() const -> Elem&
            {
                if constexpr(Dim::value == 0)
                    return *m_nativePtr;
                else
                {
                    Vec<Dim, Idx> const currentIdxDimx
                        = mapIdx<Dim::value>(Vec<DimInt<1>, Idx>{m_currentIdx}, m_extents);
                    auto const offsetInBytes = (currentIdxDimx * m_pitchBytes).sum();
                    using QualifiedByte = MimicConst<std::byte, Elem>;
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
                    // "cast from 'Byte*' to 'Elem*' increases required alignment of target type"
#    pragma GCC diagnostic ignored "-Wcast-align"
#endif
                    return *reinterpret_cast<Elem*>(reinterpret_cast<QualifiedByte*>(m_nativePtr) + offsetInBytes);
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
                }
                ALPAKA_UNREACHABLE(*m_nativePtr);
            }

        private:
            Elem* m_nativePtr;
            Idx m_currentIdx;
            Vec<Dim, Idx> m_extents;
            Vec<Dim, Idx> m_pitchBytes;
        };

        template<typename TView, typename TSfinae = void>
        struct Begin
        {
            ALPAKA_FN_HOST static auto begin(TView& view) -> IteratorView<TView>
            {
                return IteratorView<TView>(view);
            }
        };

        template<typename TView, typename TSfinae = void>
        struct End
        {
            ALPAKA_FN_HOST static auto end(TView& view) -> IteratorView<TView>
            {
                auto extents = getExtents(view);
                return IteratorView<TView>(view, extents.prod());
            }
        };
    } // namespace trait

    template<typename TView>
    using Iterator = trait::IteratorView<TView>;

    template<typename TView>
    ALPAKA_FN_HOST auto begin(TView& view) -> Iterator<TView>
    {
        return trait::Begin<TView>::begin(view);
    }

    template<typename TView>
    ALPAKA_FN_HOST auto end(TView& view) -> Iterator<TView>
    {
        return trait::End<TView>::end(view);
    }
} // namespace alpaka::test
