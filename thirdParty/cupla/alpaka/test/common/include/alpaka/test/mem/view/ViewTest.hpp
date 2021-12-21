/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/mem/view/Iterator.hpp>

#include <catch2/catch.hpp>

#include <numeric>
#include <type_traits>


namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    namespace test
    {
        //-----------------------------------------------------------------------------
        template<typename TElem, typename TDim, typename TIdx, typename TDev, typename TView>
        ALPAKA_FN_HOST auto testViewImmutable(
            TView const& view,
            TDev const& dev,
            alpaka::Vec<TDim, TIdx> const& extent,
            alpaka::Vec<TDim, TIdx> const& offset) -> void
        {
            //-----------------------------------------------------------------------------
            // alpaka::traits::DevType
            {
                static_assert(
                    std::is_same<alpaka::Dev<TView>, TDev>::value,
                    "The device type of the view has to be equal to the specified one.");
            }

            //-----------------------------------------------------------------------------
            // alpaka::traits::GetDev
            {
                REQUIRE(dev == alpaka::getDev(view));
            }

            //-----------------------------------------------------------------------------
            // alpaka::traits::DimType
            {
                static_assert(
                    alpaka::Dim<TView>::value == TDim::value,
                    "The dimensionality of the view has to be equal to the specified one.");
            }

            //-----------------------------------------------------------------------------
            // alpaka::traits::ElemType
            {
                static_assert(
                    std::is_same<alpaka::Elem<TView>, TElem>::value,
                    "The element type of the view has to be equal to the specified one.");
            }

            //-----------------------------------------------------------------------------
            // alpaka::extent::traits::GetExtent
            {
                REQUIRE(extent == alpaka::extent::getExtentVec(view));
            }

            //-----------------------------------------------------------------------------
            // alpaka::traits::GetPitchBytes
            {
                // The pitches have to be at least as large as the values we calculate here.
                auto pitchMinimum(alpaka::Vec<alpaka::DimInt<TDim::value + 1u>, TIdx>::ones());
                // Initialize the pitch between two elements of the X dimension ...
                pitchMinimum[TDim::value] = sizeof(TElem);
                // ... and fill all the other dimensions.
                for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
                {
                    pitchMinimum[i - 1] = extent[i - 1] * pitchMinimum[i];
                }

                auto const pitchView(alpaka::getPitchBytesVec(view));

                for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
                {
                    REQUIRE(pitchView[i - 1] >= pitchMinimum[i - 1]);
                }
            }

            //-----------------------------------------------------------------------------
            // alpaka::traits::GetPtrNative
            {
                // The view is a const& so the pointer has to point to a const value.
                using NativePtr = decltype(alpaka::getPtrNative(view));
                static_assert(
                    std::is_pointer<NativePtr>::value,
                    "The value returned by getPtrNative has to be a pointer.");
                static_assert(
                    std::is_const<std::remove_pointer_t<NativePtr>>::value,
                    "The value returned by getPtrNative has to be const when the view is const.");

                if(alpaka::extent::getExtentProduct(view) != static_cast<TIdx>(0u))
                {
                    // The pointer is only required to be non-null when the extent is > 0.
                    TElem const* const invalidPtr(nullptr);
                    REQUIRE(invalidPtr != alpaka::getPtrNative(view));
                }
                else
                {
                    // When the extent is 0, the pointer is undefined but it should still be possible get it.
                    alpaka::getPtrNative(view);
                }
            }

            //-----------------------------------------------------------------------------
            // alpaka::traits::GetOffset
            {
                REQUIRE(offset == alpaka::getOffsetVec(view));
            }

            //-----------------------------------------------------------------------------
            // alpaka::traits::IdxType
            {
                static_assert(
                    std::is_same<alpaka::Idx<TView>, TIdx>::value,
                    "The idx type of the view has to be equal to the specified one.");
            }
        }

        //#############################################################################
        //! Compares element-wise that all bytes are set to the same value.
        struct VerifyBytesSetKernel
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TAcc, typename TIter>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                bool* success,
                TIter const& begin,
                TIter const& end,
                std::uint8_t const& byte) const
            {
                alpaka::ignore_unused(acc);

                constexpr auto elemSizeInByte = sizeof(decltype(*begin));
                for(auto it = begin; it != end; ++it)
                {
                    auto const& elem = *it;
                    auto const pBytes = reinterpret_cast<std::uint8_t const*>(&elem);
                    for(std::size_t i = 0u; i < elemSizeInByte; ++i)
                    {
                        ALPAKA_CHECK(*success, pBytes[i] == byte);
                    }
                }
            }
        };
        //-----------------------------------------------------------------------------
        template<typename TAcc, typename TView>
        ALPAKA_FN_HOST auto verifyBytesSet(TView const& view, std::uint8_t const& byte) -> void
        {
            using Dim = alpaka::Dim<TView>;
            using Idx = alpaka::Idx<TView>;

            alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

            VerifyBytesSetKernel verifyBytesSet;

            REQUIRE(fixture(verifyBytesSet, alpaka::test::begin(view), alpaka::test::end(view), byte));
        }

        //#############################################################################
        //! Compares iterators element-wise
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal" // "comparing floating point with == or != is unsafe"
#endif
        struct VerifyViewsEqualKernel
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TAcc, typename TIterA, typename TIterB>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                bool* success,
                TIterA beginA,
                TIterA const& endA,
                TIterB beginB) const
            {
                alpaka::ignore_unused(acc);

                for(; beginA != endA; ++beginA, ++beginB)
                {
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wfloat-equal" // "comparing floating point with == or != is unsafe"
#endif
                    ALPAKA_CHECK(*success, *beginA == *beginB);
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
                }
            }
        };
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

        //-----------------------------------------------------------------------------
        template<typename TAcc, typename TViewB, typename TViewA>
        ALPAKA_FN_HOST auto verifyViewsEqual(TViewA const& viewA, TViewB const& viewB) -> void
        {
            using DimA = alpaka::Dim<TViewA>;
            using DimB = alpaka::Dim<TViewB>;
            static_assert(DimA::value == DimB::value, "viewA and viewB are required to have identical Dim");
            using IdxA = alpaka::Idx<TViewA>;
            using IdxB = alpaka::Idx<TViewB>;
            static_assert(std::is_same<IdxA, IdxB>::value, "viewA and viewB are required to have identical Idx");

            alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<DimA, IdxA>::ones());

            VerifyViewsEqualKernel verifyViewsEqualKernel;

            REQUIRE(fixture(
                verifyViewsEqualKernel,
                alpaka::test::begin(viewA),
                alpaka::test::end(viewA),
                alpaka::test::begin(viewB)));
        }

        //-----------------------------------------------------------------------------
        //! Fills the given view with increasing values starting at 0.
        template<typename TView, typename TQueue>
        ALPAKA_FN_HOST auto iotaFillView(TQueue& queue, TView& view) -> void
        {
            using Dim = alpaka::Dim<TView>;
            using Idx = alpaka::Idx<TView>;

            using DevHost = alpaka::DevCpu;
            using PltfHost = alpaka::Pltf<DevHost>;

            using Elem = alpaka::Elem<TView>;

            using ViewPlainPtr = alpaka::ViewPlainPtr<DevHost, Elem, Dim, Idx>;

            DevHost const devHost(alpaka::getDevByIdx<PltfHost>(0));

            auto const extent(alpaka::extent::getExtentVec(view));

            // Init buf with increasing values
            std::vector<Elem> v(static_cast<std::size_t>(extent.prod()), static_cast<Elem>(0));
            std::iota(v.begin(), v.end(), static_cast<Elem>(0));
            ViewPlainPtr plainBuf(v.data(), devHost, extent);

            // Copy the generated content into the given view.
            alpaka::memcpy(queue, view, plainBuf, extent);

            alpaka::wait(queue);
        }

        //-----------------------------------------------------------------------------
        template<typename TAcc, typename TView, typename TQueue>
        ALPAKA_FN_HOST auto testViewMutable(TQueue& queue, TView& view) -> void
        {
            //-----------------------------------------------------------------------------
            // alpaka::traits::GetPtrNative
            {
                // The view is a non-const so the pointer has to point to a non-const value.
                using NativePtr = decltype(alpaka::getPtrNative(view));
                static_assert(
                    std::is_pointer<NativePtr>::value,
                    "The value returned by getPtrNative has to be a pointer.");
                static_assert(
                    !std::is_const<std::remove_pointer_t<NativePtr>>::value,
                    "The value returned by getPtrNative has to be non-const when the view is non-const.");
            }

            auto const extent(alpaka::extent::getExtentVec(view));

            //-----------------------------------------------------------------------------
            // alpaka::set
            {
                std::uint8_t const byte(static_cast<uint8_t>(42u));
                alpaka::memset(queue, view, byte, extent);
                alpaka::wait(queue);
                verifyBytesSet<TAcc>(view, byte);
            }

            //-----------------------------------------------------------------------------
            // alpaka::copy
            {
                using Elem = alpaka::Elem<TView>;
                using Idx = alpaka::Idx<TView>;

                auto const devAcc = alpaka::getDev(view);

                //-----------------------------------------------------------------------------
                // alpaka::copy into given view
                {
                    auto srcBufAcc(alpaka::allocBuf<Elem, Idx>(devAcc, extent));
                    iotaFillView(queue, srcBufAcc);
                    alpaka::memcpy(queue, view, srcBufAcc, extent);
                    alpaka::wait(queue);
                    verifyViewsEqual<TAcc>(view, srcBufAcc);
                }

                //-----------------------------------------------------------------------------
                // alpaka::copy from given view
                {
                    auto dstBufAcc(alpaka::allocBuf<Elem, Idx>(devAcc, extent));
                    alpaka::memcpy(queue, dstBufAcc, view, extent);
                    alpaka::wait(queue);
                    verifyViewsEqual<TAcc>(dstBufAcc, view);
                }
            }
        }
    } // namespace test
} // namespace alpaka
