/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace alpaka::test
{
    namespace trait
    {
        // \tparam T Type to conditionally make const.
        // \tparam TSource Type to mimic the constness of.
        template<typename T, typename TSource>
        using MimicConst = std::conditional_t<std::is_const_v<TSource>, std::add_const_t<T>, std::remove_const_t<T>>;

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'Byte*' to 'Elem*' increases required alignment of target type"
#endif
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
                , m_extents(getExtentVec(view))
                , m_pitchBytes(getPitchBytesVec(view))
            {
            }

            ALPAKA_FN_HOST IteratorView(TView& view) : IteratorView(view, 0)
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
                    using Dim1 = DimInt<1>;
                    using DimMin1 = DimInt<Dim::value - 1u>;

                    Vec<Dim1, Idx> const currentIdxDim1{m_currentIdx};
                    Vec<Dim, Idx> const currentIdxDimx(mapIdx<Dim::value>(currentIdxDim1, m_extents));

                    // [pz, py, px] -> [py, px]
                    auto const pitchWithoutOutermost = subVecEnd<DimMin1>(m_pitchBytes);
                    // [ElemSize]
                    Vec<Dim1, Idx> const elementSizeVec = static_cast<Idx>(sizeof(Elem));
                    // [py, px] ++ [ElemSize] -> [py, px, ElemSize]
                    Vec<Dim, Idx> const dstPitchBytes = concatVec(pitchWithoutOutermost, elementSizeVec);
                    // [py, px, ElemSize] [z, y, x] -> [py*z, px*y, ElemSize*x]
                    auto const dimensionalOffsetsInByte = currentIdxDimx * dstPitchBytes;
                    // sum{[py*z, px*y, ElemSize*x]} -> offset in byte
                    auto const offsetInByte = dimensionalOffsetsInByte.foldrAll(std::plus<Idx>());

                    using Byte = MimicConst<std::uint8_t, Elem>;
                    Byte* ptr(reinterpret_cast<Byte*>(m_nativePtr) + offsetInByte);

#if 0
                    std::cout
                        << " i1: " << currentIdxDim1
                        << " in: " << currentIdxDimx
                        << " dpb: " << dstPitchBytes
                        << " offb: " << offsetInByte
                        << " ptr: " << reinterpret_cast<void const *>(ptr)
                        << " v: " << *reinterpret_cast<Elem *>(ptr)
                        << std::endl;
#endif
                    return *reinterpret_cast<Elem*>(ptr);
                }
                ALPAKA_UNREACHABLE(*m_nativePtr);
            }

        private:
            Elem* m_nativePtr;
            Idx m_currentIdx;
            Vec<Dim, Idx> m_extents;
            Vec<Dim, Idx> m_pitchBytes;
        };
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

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
                auto extents = getExtentVec(view);
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
