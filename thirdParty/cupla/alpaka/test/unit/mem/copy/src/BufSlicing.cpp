/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Jakob Krude, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/DemangleTypeNames.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/Iterator.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(push)
#    pragma warning(disable : 4127) // suppress warning for c++17 conditional expression is constant
#endif

template<typename TDim, typename TIdx, typename TAcc, typename TData, typename Vec = alpaka::Vec<TDim, TIdx>>
struct TestContainer
{
    using AccQueueProperty = alpaka::Blocking;
    using DevQueue = alpaka::Queue<TAcc, AccQueueProperty>;
    using DevAcc = alpaka::Dev<TAcc>;
    using PlatformAcc = alpaka::Platform<TAcc>;

    using DevHost = alpaka::DevCpu;
    using PlatformHost = alpaka::Platform<DevHost>;

    using BufHost = alpaka::Buf<DevHost, TData, TDim, TIdx>;
    using BufDevice = alpaka::Buf<DevAcc, TData, TDim, TIdx>;

    using SubView = alpaka::ViewSubView<DevAcc, TData, TDim, TIdx>;

    PlatformAcc platformAcc{};
    DevAcc devAcc{alpaka::getDevByIdx(platformAcc, 0)};
    PlatformHost platformHost{};
    DevHost devHost{alpaka::getDevByIdx(platformHost, 0u)};
    DevQueue devQueue{devAcc};

    auto createHostBuffer(Vec extents, bool indexed) -> BufHost
    {
        BufHost bufHost(alpaka::allocBuf<TData, TIdx>(devHost, extents));
        if(indexed)
        {
            TData* const ptr = bufHost.data();
            for(TIdx i = 0; i < extents.prod(); ++i)
            {
                ptr[i] = static_cast<TData>(i);
            }
        }
        return bufHost;
    }

    auto createDeviceBuffer(Vec extents) -> BufDevice
    {
        BufDevice bufDevice(alpaka::allocBuf<TData, TIdx>(devAcc, extents));
        return bufDevice;
    }

    auto copyToAcc(BufHost bufHost, BufDevice bufAcc, Vec extents) -> void
    {
        alpaka::memcpy(devQueue, bufAcc, bufHost, extents);
    }

    auto copyToHost(BufDevice bufAcc, BufHost bufHost, Vec extents) -> void
    {
        alpaka::memcpy(devQueue, bufHost, bufAcc, extents);
    }

    auto copySliceOnDevice(BufDevice bufferToBeSliced, Vec subViewExtents, Vec offsets) -> BufDevice
    {
        BufDevice slicedBuffer = createDeviceBuffer(subViewExtents);
        // Create a subView with a possible offset.
        SubView subView = SubView(bufferToBeSliced, subViewExtents, offsets);
        // Copy the subView into a new buffer.
        alpaka::memcpy(devQueue, slicedBuffer, subView, subViewExtents);
        return slicedBuffer;
    }

    auto zeroSliceOnDevice(BufDevice bufferToBeSliced, Vec subViewExtents, Vec offsets) -> void
    {
        // Create a subView with a possible offset.
        SubView subView = SubView(bufferToBeSliced, subViewExtents, offsets);
        // Fill the subView with zeros.
        alpaka::memset(devQueue, subView, 0u);
    }

    auto compareBuffer(BufHost const& bufferA, BufHost const& bufferB, Vec const& extents) const
    {
        TData const* const ptrA = bufferA.data();
        TData const* const ptrB = bufferB.data();
        for(TIdx i = 0; i < extents.prod(); ++i)
        {
            INFO("Dim: " << TDim::value);
            INFO("Idx: " << alpaka::core::demangled<TIdx>);
            INFO("Acc: " << alpaka::trait::GetAccName<TAcc>::getAccName());
            INFO("i: " << i);
            REQUIRE(ptrA[i] == Catch::Approx(ptrB[i]));
        }
    }
};

using DataTypes = std::tuple<int, float, double>;

using TestAccWithDataTypes = alpaka::meta::CartesianProduct<std::tuple, alpaka::test::TestAccs, DataTypes>;

TEMPLATE_LIST_TEST_CASE("memBufSlicingMemcpyTest", "[memBuf]", TestAccWithDataTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using Data = std::tuple_element_t<1, TestType>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    // Test only buffer slices with up to three dimensions.
    if constexpr(Dim::value < 4)
    {
        TestContainer<Dim, Idx, Acc, Data> slicingTest;

        auto const extents = alpaka::test::extentBuf<Dim, Idx>;
        auto const extentsSubView = alpaka::test::extentSubView<Dim, Idx>;
        auto const offsets = alpaka::test::offset<Dim, Idx>;

        // This is the initial buffer.
        auto const indexedBuffer = slicingTest.createHostBuffer(extents, true);

        // This buffer will hold the copy of the initial buffer on the device.
        auto deviceBuffer = slicingTest.createDeviceBuffer(extents);

        // This buffer will hold the sliced-buffer when it is copied back to the host.
        auto resultBuffer = slicingTest.createHostBuffer(extentsSubView, false);

        // Copy the initial buffer to the device.
        slicingTest.copyToAcc(indexedBuffer, deviceBuffer, extents);

        // Make a copy of a slice of the buffer on the device.
        auto slicedBuffer = slicingTest.copySliceOnDevice(deviceBuffer, extentsSubView, offsets);

        // Copy the slice back to the host.
        slicingTest.copyToHost(slicedBuffer, resultBuffer, extentsSubView);

        // Compute the expected content of the slice.
        using Dim1 = alpaka::DimInt<1u>;
        auto correctResults = slicingTest.createHostBuffer(extentsSubView, false);
        Data* const ptr = correctResults.data();
        for(Idx i = 0; i < extentsSubView.prod(); ++i)
        {
            auto mappedToND = alpaka::mapIdx<Dim::value, Dim1::value>(alpaka::Vec<Dim1, Idx>(i), extentsSubView);
            auto addedOffset = mappedToND + offsets;
            auto mappedTo1D = alpaka::mapIdx<Dim1::value>(
                addedOffset,
                extents)[0]; // take the only element in the vector
            ptr[i] = static_cast<Data>(mappedTo1D);
        }

        // Compare the resultBuffer with the results computed manually.
        slicingTest.compareBuffer(resultBuffer, correctResults, extentsSubView);
    }
}

TEMPLATE_LIST_TEST_CASE("memBufSlicingMemsetTest", "[memBuf]", TestAccWithDataTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using Data = std::tuple_element_t<1, TestType>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    // Test only buffer slices with up to three dimensions.
    if constexpr(Dim::value < 4)
    {
        TestContainer<Dim, Idx, Acc, Data> slicingTest;

        auto const extents = alpaka::test::extentBuf<Dim, Idx>;
        auto const extentsSubView = alpaka::test::extentSubView<Dim, Idx>;
        auto const offsets = alpaka::test::offset<Dim, Idx>;

        // This is the initial buffer.
        auto const indexedBuffer = slicingTest.createHostBuffer(extents, true);

        // This buffer will hold the copy of the initial buffer on the device.
        auto deviceBuffer = slicingTest.createDeviceBuffer(extents);

        // This buffer will hold a copy of the initial buffer, after a slice has been set to zeroes.
        auto resultBuffer = slicingTest.createHostBuffer(extents, false);

        // Copy the initial buffer to the device.
        slicingTest.copyToAcc(indexedBuffer, deviceBuffer, extents);

        // Fill a slice of the buffer with zeroes.
        slicingTest.zeroSliceOnDevice(deviceBuffer, extentsSubView, offsets);

        // Copy the buffer back to the host
        slicingTest.copyToHost(deviceBuffer, resultBuffer, extents);

        // Compute the expected content of the buffer, with a slice set to zeroes.
        using Dim1 = alpaka::DimInt<1u>;
        auto correctResults = slicingTest.createHostBuffer(extents, true);
        Data* const ptr = correctResults.data();
        for(Idx i = 0; i < extents.prod(); ++i)
        {
            auto mappedToND = alpaka::mapIdx<Dim::value, Dim1::value>(alpaka::Vec<Dim1, Idx>(i), extents);
            if((mappedToND >= offsets && mappedToND < offsets + extentsSubView).all())
                ptr[i] = static_cast<Data>(0u);
        }

        // Compare the resultBuffer with the results computed manually.
        slicingTest.compareBuffer(resultBuffer, correctResults, extentsSubView);
    }
}

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif
