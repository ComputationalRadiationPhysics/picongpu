/**
 * \file
 * Copyright 2015-2017 Benjamin Worpitz
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

#include <alpaka/test/mem/view/Iterator.hpp>

#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

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
            namespace view
            {
                //-----------------------------------------------------------------------------
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize,
                    typename TDev,
                    typename TView>
                static auto viewTestImmutable(
                    TView const & view,
                    TDev const & dev,
                    alpaka::vec::Vec<TDim, TSize> const & extent,
                    alpaka::vec::Vec<TDim, TSize> const & offset)
                -> void
                {
                    //-----------------------------------------------------------------------------
                    // alpaka::dev::traits::DevType
                    {
                        static_assert(
                            std::is_same<alpaka::dev::Dev<TView>, TDev>::value,
                            "The device type of the view has to be equal to the specified one.");
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::dev::traits::GetDev
                    {
                        BOOST_REQUIRE(
                            dev == alpaka::dev::getDev(view));
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::dim::traits::DimType
                    {
                        static_assert(
                            alpaka::dim::Dim<TView>::value == TDim::value,
                            "The dimensionality of the view has to be equal to the specified one.");
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::elem::traits::ElemType
                    {
                        static_assert(
                            std::is_same<alpaka::elem::Elem<TView>, TElem>::value,
                            "The element type of the view has to be equal to the specified one.");
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::extent::traits::GetExtent
                    {
                        BOOST_REQUIRE_EQUAL(
                            extent,
                            alpaka::extent::getExtentVec(view));
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::mem::view::traits::GetPitchBytes
                    {
                        // The pitches have to be at least as large as the values we calculate here.
                        auto pitchMinimum(alpaka::vec::Vec<alpaka::dim::DimInt<TDim::value + 1u>, TSize>::ones());
                        // Initialize the pitch between two elements of the X dimension ...
                        pitchMinimum[TDim::value] = sizeof(TElem);
                        // ... and fill all the other dimensions.
                        for(TSize i = TDim::value; i > static_cast<TSize>(0u); --i)
                        {
                            pitchMinimum[i-1] = extent[i-1] * pitchMinimum[i];
                        }

                        auto const pitchView(alpaka::mem::view::getPitchBytesVec(view));

                        for(TSize i = TDim::value; i > static_cast<TSize>(0u); --i)
                        {
                            BOOST_REQUIRE_GE(
                                pitchView[i-1],
                                pitchMinimum[i-1]);
                        }
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::mem::view::traits::GetPtrNative
                    {
                        if(alpaka::extent::getProductOfExtent(view) != static_cast<TSize>(0u))
                        {
                            // The pointer is only required to be non-null when the extent is > 0.
                            TElem const * const invalidPtr(nullptr);
                            BOOST_REQUIRE_NE(
                                invalidPtr,
                                alpaka::mem::view::getPtrNative(view));
                        }
                        else
                        {
                            // When the extent is 0, the pointer is undefined but it should still be possible get it.
                            alpaka::mem::view::getPtrNative(view);
                        }
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::offset::traits::GetOffset
                    {
                        BOOST_REQUIRE_EQUAL(
                            offset,
                            alpaka::offset::getOffsetVec(view));
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::size::traits::SizeType
                    {
                        static_assert(
                            std::is_same<alpaka::size::Size<TView>, TSize>::value,
                            "The size type of the view has to be equal to the specified one.");
                    }
                }

                //#############################################################################
                //! Compares element-wise that all bytes are set to the same value.
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wfloat-equal"  // "comparing floating point with == or != is unsafe"
#endif
                struct VerifyBytesSetKernel
                {
                    ALPAKA_NO_HOST_ACC_WARNING
                    template<
                        typename TAcc,
                        typename TIter>
                    ALPAKA_FN_ACC void operator()(
                        TAcc const & acc,
                        TIter const & begin,
                        TIter const & end,
                        std::uint8_t const & byte) const
                    {
                        constexpr auto elemSizeInByte = sizeof(decltype(*begin));
                        (void)acc;
                        for(auto it = begin; it != end; ++it)
                        {
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wfloat-equal" // "comparing floating point with == or != is unsafe"
#endif
                            auto const& elem = *it;
                            auto const pBytes = reinterpret_cast<std::uint8_t const *>(&elem);
                            for(std::size_t i = 0u; i < elemSizeInByte; ++i)
                            {
                                BOOST_VERIFY(pBytes[i] == byte);
                            }
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                        }
                    }
                };
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif
                //-----------------------------------------------------------------------------
                template<
                    typename TAcc,
                    typename TView,
                    typename TStream>
                static auto verifyBytesSet(
                    TStream & stream,
                    TView const & view,
                    std::uint8_t const & byte)
                -> void
                {
                    using Dim = alpaka::dim::Dim<TView>;
                    using Size = alpaka::size::Size<TView>;

                    using Vec = alpaka::vec::Vec<Dim, Size>;
                    auto const elementsPerThread(Vec::ones());
                    auto const threadsPerBlock(Vec::ones());
                    auto const blocksPerGrid(Vec::ones());

                    auto const workdiv(
                        alpaka::workdiv::WorkDivMembers<Dim, Size>(
                            blocksPerGrid,
                            threadsPerBlock,
                            elementsPerThread));
                    VerifyBytesSetKernel verifyBytesSet;
                    auto const compare(
                        alpaka::exec::create<TAcc>(
                            workdiv,
                            verifyBytesSet,
                            alpaka::test::mem::view::begin(view),
                            alpaka::test::mem::view::end(view),
                            byte));
                    alpaka::stream::enqueue(stream, compare);
                    alpaka::wait::wait(stream);
                }

                //#############################################################################
                //! Compares iterators element-wise
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wfloat-equal"  // "comparing floating point with == or != is unsafe"
#endif
                struct VerifyViewsEqualKernel
                {
                    ALPAKA_NO_HOST_ACC_WARNING
                    template<
                        typename TAcc,
                        typename TIterA,
                        typename TIterB>
                    ALPAKA_FN_ACC void operator()(
                        TAcc const & acc,
                        TIterA beginA,
                        TIterA const & endA,
                        TIterB beginB) const
                    {
                        (void)acc;
                        for(; beginA != endA; ++beginA, ++beginB)
                        {
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wfloat-equal" // "comparing floating point with == or != is unsafe"
#endif
                            BOOST_VERIFY(*beginA == *beginB);
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                        }
                    }
                };
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif

                //-----------------------------------------------------------------------------
                template<
                    typename TAcc,
                    typename TViewB,
                    typename TViewA,
                    typename TStream>
                static auto verifyViewsEqual(
                    TStream & stream,
                    TViewA const & viewA,
                    TViewB const & viewB)
                -> void
                {
                    using DimA = alpaka::dim::Dim<TViewA>;
                    using DimB = alpaka::dim::Dim<TViewB>;
                    static_assert(DimA::value == DimB::value, "viewA and viewB are required to have identical Dim");
                    using SizeA = alpaka::size::Size<TViewA>;
                    using SizeB = alpaka::size::Size<TViewB>;
                    static_assert(std::is_same<SizeA, SizeB>::value, "viewA and viewB are required to have identical Size");

                    using Vec = alpaka::vec::Vec<DimA, SizeA>;
                    auto const elementsPerThread(Vec::ones());
                    auto const threadsPerBlock(Vec::ones());
                    auto const blocksPerGrid(Vec::ones());

                    auto const workdiv(
                        alpaka::workdiv::WorkDivMembers<DimA, SizeA>(
                            blocksPerGrid,
                            threadsPerBlock,
                            elementsPerThread));
                    VerifyViewsEqualKernel verifyViewsEqualKernel;
                    auto const compare(
                        alpaka::exec::create<TAcc>(
                            workdiv,
                            verifyViewsEqualKernel,
                            alpaka::test::mem::view::begin(viewA),
                            alpaka::test::mem::view::end(viewA),
                            alpaka::test::mem::view::begin(viewB)));
                    alpaka::stream::enqueue(stream, compare);
                    alpaka::wait::wait(stream);
                }

                //-----------------------------------------------------------------------------
                //! Fills the given view with increasing values starting at 0.
                template<
                    typename TView,
                    typename TStream>
                static auto iotaFillView(
                    TStream & stream,
                    TView & view)
                -> void
                {
                    using Dim = alpaka::dim::Dim<TView>;
                    using Size = alpaka::size::Size<TView>;

                    using DevHost = alpaka::dev::DevCpu;
                    using PltfHost = alpaka::pltf::Pltf<DevHost>;

                    using Elem = alpaka::elem::Elem<TView>;

                    using ViewPlainPtr = alpaka::mem::view::ViewPlainPtr<DevHost, Elem, Dim, Size>;

                    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0));

                    auto const extent(alpaka::extent::getExtentVec(view));

                    // Init buf with increasing values
                    std::vector<Elem> v(static_cast<std::size_t>(extent.prod()), static_cast<Elem>(0));
                    std::iota(v.begin(), v.end(), static_cast<Elem>(0));
                    ViewPlainPtr plainBuf(v.data(), devHost, extent);

                    // Copy the generated content into the given view.
                    alpaka::mem::view::copy(stream, view, plainBuf, extent);

                    alpaka::wait::wait(stream);
                }

                //-----------------------------------------------------------------------------
                template<
                    typename TAcc,
                    typename TView,
                    typename TStream>
                static auto viewTestMutable(
                    TStream & stream,
                    TView & view)
                -> void
                {
                    using Size = alpaka::size::Size<TView>;

                    auto const extent(alpaka::extent::getExtentVec(view));

                    //-----------------------------------------------------------------------------
                    // alpaka::mem::view::set
                    {
                        std::uint8_t const byte(static_cast<uint8_t>(42u));
                        alpaka::mem::view::set(stream, view, byte, extent);
                        verifyBytesSet<TAcc>(stream, view, byte);
                    }

                    //-----------------------------------------------------------------------------
                    // alpaka::mem::view::copy
                    {
                        using Elem = alpaka::elem::Elem<TView>;

                        auto const devAcc = alpaka::dev::getDev(view);

                        //-----------------------------------------------------------------------------
                        // alpaka::mem::view::copy into given view
                        {
                            auto srcBufAcc(alpaka::mem::buf::alloc<Elem, Size>(devAcc, extent));
                            iotaFillView(stream, srcBufAcc);
                            alpaka::mem::view::copy(stream, view, srcBufAcc, extent);
                            alpaka::test::mem::view::verifyViewsEqual<TAcc>(stream, view, srcBufAcc);
                        }

                        //-----------------------------------------------------------------------------
                        // alpaka::mem::view::copy from given view
                        {
                            auto dstBufAcc(alpaka::mem::buf::alloc<Elem, Size>(devAcc, extent));
                            alpaka::mem::view::copy(stream, dstBufAcc, view, extent);
                            alpaka::test::mem::view::verifyViewsEqual<TAcc>(stream, dstBufAcc, view);
                        }
                    }
                }
            }
        }
    }
}
