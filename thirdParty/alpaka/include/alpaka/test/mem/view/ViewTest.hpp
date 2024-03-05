/* Copyright 2023 Benjamin Worpitz, Sergei Bastrakov, René Widera, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/alpaka.hpp"
#include "alpaka/test/KernelExecutionFixture.hpp"
#include "alpaka/test/mem/view/Iterator.hpp"

#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

//! The test specifics.
namespace alpaka::test
{
    template<typename TElem, typename TDim, typename TIdx, typename TDev, typename TView>
    ALPAKA_FN_HOST auto testViewImmutable(
        TView const& view,
        TDev const& dev,
        Vec<TDim, TIdx> const& extent,
        Vec<TDim, TIdx> const& offset) -> void
    {
        // trait::DevType
        {
            static_assert(
                std::is_same_v<Dev<TView>, TDev>,
                "The device type of the view has to be equal to the specified one.");
        }

        // trait::GetDev
        {
            REQUIRE(dev == getDev(view));
        }

        // trait::DimType
        {
            static_assert(
                Dim<TView>::value == TDim::value,
                "The dimensionality of the view has to be equal to the specified one.");
        }

        // trait::ElemType
        {
            static_assert(
                std::is_same_v<Elem<TView>, TElem>,
                "The element type of the view has to be equal to the specified one.");
        }

        // trait::GetExtents
        {
            REQUIRE(extent == getExtents(view));
        }

        // trait::GetPitchBytes
        {
            auto const pitchMinimum = alpaka::detail::calculatePitchesFromExtents<TElem>(extent);
            auto const pitchView = getPitchesInBytes(view);

            for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
            {
                REQUIRE(pitchView[i - 1] >= pitchMinimum[i - 1]);
            }
        }

        // trait::GetPtrNative
        {
            // The view is a const& so the pointer has to point to a const value.
            using NativePtr = decltype(getPtrNative(view));
            static_assert(std::is_pointer_v<NativePtr>, "The value returned by getPtrNative has to be a pointer.");
            static_assert(
                std::is_const_v<std::remove_pointer_t<NativePtr>>,
                "The value returned by getPtrNative has to be const when the view is const.");

            if(getExtentProduct(view) != static_cast<TIdx>(0u))
            {
                // The pointer is only required to be non-null when the extent is > 0.
                TElem const* const invalidPtr(nullptr);
                REQUIRE(invalidPtr != getPtrNative(view));
            }
            else
            {
                // When the extent is 0, the pointer is undefined but it should still be possible get it.
                getPtrNative(view);
            }
        }

        // trait::GetOffsets
        {
            REQUIRE(offset == getOffsets(view));
        }

        // trait::IdxType
        {
            static_assert(
                std::is_same_v<Idx<TView>, TIdx>,
                "The idx type of the view has to be equal to the specified one.");
        }
    }

    //! Compares element-wise that all bytes are set to the same value.
    struct VerifyBytesSetKernel
    {
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TIter>
        ALPAKA_FN_ACC void operator()(
            TAcc const& acc [[maybe_unused]], // used by SYCL back-end
            bool* success,
            TIter const& begin,
            TIter const& end,
            std::uint8_t const& byte) const
        {
            constexpr auto elemSizeInByte = static_cast<unsigned>(sizeof(decltype(*begin)));
            for(auto it = begin; it != end; ++it)
            {
                auto const& elem = *it;
                auto const pBytes = reinterpret_cast<std::uint8_t const*>(&elem);
                for(unsigned i = 0; i < elemSizeInByte; ++i)
                {
                    if(pBytes[i] != byte)
                    {
                        printf("Byte at offset %u is different: %u != %u\n", i, unsigned{pBytes[i]}, unsigned{byte});
                        *success = false;
                    }
                }
            }
        }
    };

    template<typename TAcc, typename TView>
    ALPAKA_FN_HOST auto verifyBytesSet(TView const& view, std::uint8_t const& byte) -> void
    {
        using Dim = Dim<TView>;
        using Idx = Idx<TView>;

        KernelExecutionFixture<TAcc> fixture(Vec<Dim, Idx>::ones());

        VerifyBytesSetKernel verifyBytesSet;

        REQUIRE(fixture(verifyBytesSet, test::begin(view), test::end(view), byte));
    }

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
            TAcc const& acc [[maybe_unused]], // used by SYCL back-end
            bool* success,
            TIterA beginA,
            TIterA const& endA,
            TIterB beginB) const
        {
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

    template<typename TAcc, typename TViewB, typename TViewA>
    ALPAKA_FN_HOST auto verifyViewsEqual(TViewA const& viewA, TViewB const& viewB) -> void
    {
        using DimA = Dim<TViewA>;
        using DimB = Dim<TViewB>;
        static_assert(DimA::value == DimB::value, "viewA and viewB are required to have identical Dim");
        using IdxA = Idx<TViewA>;
        using IdxB = Idx<TViewB>;
        static_assert(std::is_same_v<IdxA, IdxB>, "viewA and viewB are required to have identical Idx");

        test::KernelExecutionFixture<TAcc> fixture(Vec<DimA, IdxA>::ones());

        VerifyViewsEqualKernel verifyViewsEqualKernel;

        REQUIRE(fixture(verifyViewsEqualKernel, test::begin(viewA), test::end(viewA), test::begin(viewB)));
    }

    //! Fills the given view with increasing values starting at 0.
    template<typename TView, typename TQueue>
    ALPAKA_FN_HOST auto iotaFillView(TQueue& queue, TView& view) -> void
    {
        using Elem = Elem<TView>;

        auto const platformHost = alpaka::PlatformCpu{};
        auto const devHost = alpaka::getDevByIdx(platformHost, 0);

        auto const extent = getExtents(view);

        // Init buf with increasing values
        std::vector<Elem> v(static_cast<std::size_t>(extent.prod()), static_cast<Elem>(0));
        std::iota(std::begin(v), std::end(v), static_cast<Elem>(0));
        auto plainBuf = createView(devHost, v, extent);

        // Copy the generated content into the given view.
        memcpy(queue, view, plainBuf);

        wait(queue);
    }

    template<typename TAcc, typename TView, typename TQueue>
    ALPAKA_FN_HOST auto testViewMutable(TQueue& queue, TView& view) -> void
    {
        // trait::GetPtrNative
        {
            // The view is a non-const so the pointer has to point to a non-const value.
            using NativePtr = decltype(getPtrNative(view));
            static_assert(std::is_pointer_v<NativePtr>, "The value returned by getPtrNative has to be a pointer.");
            static_assert(
                !std::is_const_v<std::remove_pointer_t<NativePtr>>,
                "The value returned by getPtrNative has to be non-const when the view is non-const.");
        }

        // set
        {
            auto const byte(static_cast<uint8_t>(42u));
            memset(queue, view, byte);
            wait(queue);
            verifyBytesSet<TAcc>(view, byte);
        }

        // copy
        {
            using Elem = Elem<TView>;
            using Idx = Idx<TView>;

            auto const devAcc = getDev(view);
            auto const extent = getExtents(view);

            // copy into given view
            {
                auto srcBufAcc = allocBuf<Elem, Idx>(devAcc, extent);
                iotaFillView(queue, srcBufAcc);
                memcpy(queue, view, srcBufAcc);
                wait(queue);
                verifyViewsEqual<TAcc>(view, srcBufAcc);
            }

            // copy from given view
            {
                auto dstBufAcc = allocBuf<Elem, Idx>(devAcc, extent);
                memcpy(queue, dstBufAcc, view);
                wait(queue);
                verifyViewsEqual<TAcc>(dstBufAcc, view);
            }
        }
    }
} // namespace alpaka::test
