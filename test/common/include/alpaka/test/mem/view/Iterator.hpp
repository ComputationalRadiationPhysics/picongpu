/**
* \file
* Copyright 2014-2017 Erik Zenker, Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    namespace test
    {
        //-----------------------------------------------------------------------------
        //! The test mem specifics.
        namespace mem
        {
            //-----------------------------------------------------------------------------
            //!
            namespace view
            {
                //-----------------------------------------------------------------------------
                //!
                namespace traits
                {
                    //#############################################################################
                    // \tparam T Type to conditionally make const.
                    // \tparam TSource Type to mimic the constness of.
                    template<
                        typename T,
                        typename TSource>
                    using MimicConst = typename std::conditional<
                        std::is_const<TSource>::value,
                        typename std::add_const<T>::type,
                        typename std::remove_const<T>::type>;

                    //#############################################################################
                    template<
                        typename TView,
                        typename TSfinae = void>
                    class IteratorView
                    {
                        using TViewDecayed = typename std::decay<TView>::type;
                        using Dim = alpaka::dim::Dim<TViewDecayed>;
                        using Size = alpaka::size::Size<TViewDecayed>;
                        using Elem = typename MimicConst<alpaka::elem::Elem<TViewDecayed>, TView>::type;

                    public:
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST IteratorView(
                            TView & view,
                            Size const idx) :
                                m_nativePtr(alpaka::mem::view::getPtrNative(view)),
                                m_currentIdx(idx),
                                m_extents(alpaka::extent::getExtentVec(view)),
                                m_pitchBytes(alpaka::mem::view::getPitchBytesVec(view))
                        {}

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST IteratorView(
                            TView & view) :
                                IteratorView(view, 0)
                        {}

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator++()
                        -> IteratorView&
                        {
                            ++m_currentIdx;
                            return *this;
                        }

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator--()
                        -> IteratorView&
                        {
                            --m_currentIdx;
                            return *this;
                        }

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator++(
                            int)
                        -> IteratorView
                        {
                            IteratorView iterCopy = *this;
                            m_currentIdx++;
                            return iterCopy;
                        }

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator--(
                            int)
                        -> IteratorView
                        {
                            IteratorView iterCopy = *this;
                            m_currentIdx--;
                            return iterCopy;
                        }

                        //-----------------------------------------------------------------------------
                        template<typename TIter>
                        ALPAKA_FN_HOST_ACC auto operator==(
                            TIter &other) const
                        -> bool
                        {
                            return m_currentIdx == other.m_currentIdx;
                        }

                        //-----------------------------------------------------------------------------
                        template<typename TIter>
                        ALPAKA_FN_HOST_ACC auto operator!=(
                            TIter &other) const
                        -> bool
                        {
                            return m_currentIdx != other.m_currentIdx;
                        }

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST_ACC auto operator*() const
                        -> Elem &
                        {
                            using Dim1 = dim::DimInt<1>;
                            using DimMin1 = dim::DimInt<Dim::value - 1u>;

                            vec::Vec<Dim1, Size> const currentIdxDim1{m_currentIdx};
                            vec::Vec<Dim, Size> const currentIdxDimx(idx::mapIdx<Dim::value>(currentIdxDim1, m_extents));

                            // [pz, py, px] -> [py, px]
                            auto const pitchWithoutOutermost(vec::subVecEnd<DimMin1>(m_pitchBytes));
                            // [ElemSize]
                            vec::Vec<Dim1, Size> const elementSizeVec(static_cast<Size>(sizeof(Elem)));
                            // [py, px] ++ [ElemSize] -> [py, px, ElemSize]
                            vec::Vec<Dim, Size> const dstPitchBytes(vec::concat(pitchWithoutOutermost, elementSizeVec));
                            // [py, px, ElemSize] [z, y, x] -> [py*z, px*y, ElemSize*x]
                            auto const dimensionalOffsetsInByte(currentIdxDimx * dstPitchBytes);
                            // sum{[py*z, px*y, ElemSize*x]} -> offset in byte
                            auto const offsetInByte(dimensionalOffsetsInByte.foldrAll(
                                [](Size a, Size b)
                                {
                                    return static_cast<Size>(a + b);
                                }));

                            using Byte = typename MimicConst<std::uint8_t, Elem>::type;
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

                            return *reinterpret_cast<Elem *>(ptr);
                        }

                    private:
                        Elem * const m_nativePtr;
                        Size m_currentIdx;
                        vec::Vec<Dim, Size> const m_extents;
                        vec::Vec<Dim, Size> const m_pitchBytes;
                    };

                    //#############################################################################
                    template<
                        typename TView,
                        typename TSfinae = void>
                    struct Begin
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST static auto begin(
                            TView & view)
                        -> IteratorView<TView>
                        {
                            return IteratorView<TView>(view);
                        }
                    };

                    //#############################################################################
                    template<
                        typename TView,
                        typename TSfinae = void>
                    struct End
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST static auto end(
                            TView & view)
                        -> IteratorView<TView>
                        {
                            auto extents = alpaka::extent::getExtentVec(view);
                            return IteratorView<TView>(view, extents.prod());
                        }
                    };
                }

                //#############################################################################
                template<
                    typename TView>
                using Iterator = traits::IteratorView<TView>;

                //-----------------------------------------------------------------------------
                template<
                    typename TView>
                ALPAKA_FN_HOST static auto begin(
                    TView & view)
                -> Iterator<TView>
                {
                    return traits::Begin<TView>::begin(view);
                }

                //-----------------------------------------------------------------------------
                template<
                    typename TView>
                ALPAKA_FN_HOST static auto end(
                    TView & view)
                -> Iterator<TView>
                {
                    return traits::End<TView>::end(view);
                }
            }
        }
    }
}
