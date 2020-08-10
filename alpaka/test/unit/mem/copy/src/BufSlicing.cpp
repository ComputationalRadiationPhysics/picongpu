/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Jakob Krude
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/Iterator.hpp>
#include <alpaka/test/Extent.hpp>
#include <catch2/catch.hpp>

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
    #pragma warning(push)
    #pragma warning(disable: 4127) // suppress warning for c++17 conditional expression is constant
#endif

template<
    typename TDim,
    typename TIdx,
    typename TAcc,
    typename TData,
    typename Vec = alpaka::vec::Vec<
        TDim,
        TIdx>>
struct TestContainer
{

    using AccQueueProperty = alpaka::queue::Blocking;
    using DevQueue = alpaka::queue::Queue<
        TAcc,
        AccQueueProperty
    >;
    using DevAcc = alpaka::dev::Dev<TAcc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;

    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;

    using BufHost = alpaka::mem::buf::Buf<
        DevHost,
        TData,
        TDim,
        TIdx
    >;
    using BufDevice = alpaka::mem::buf::Buf<
        DevAcc,
        TData,
        TDim,
        TIdx
    >;

    using SubView = alpaka::mem::view::ViewSubView<
        DevAcc,
        TData,
        TDim,
        TIdx
    >;

    DevAcc const devAcc;
    DevHost const devHost;
    DevQueue devQueue;


    // Constructor
    TestContainer():
        devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u)),
        devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u)),
        devQueue(devAcc)
    {}


    auto createHostBuffer(
        Vec extents,
        bool indexed
    ) -> BufHost
    {
        BufHost bufHost(
            alpaka::mem::buf::alloc<
                TData,
                TIdx
            >(
                devHost,
                extents
            ));
        if(indexed)
        {
            TData *const ptr = alpaka::mem::view::getPtrNative(bufHost);
            for(TIdx i(0);i < extents.prod();++i)
            {
                ptr[i] = static_cast<TData>(i);
            }
        }
        return bufHost;
    }


    auto createDeviceBuffer(Vec extents) -> BufDevice
    {
        BufDevice bufDevice(
            alpaka::mem::buf::alloc<
                TData,
                TIdx
            >(
                devAcc,
                extents
            ));
        return bufDevice;
    }


    auto copyToAcc(
        BufHost bufHost,
        BufDevice bufAcc,
        Vec extents
    ) -> void
    {
        alpaka::mem::view::copy(
            devQueue,
            bufAcc,
            bufHost,
            extents
        );
    }


    auto copyToHost(
        BufDevice bufAcc,
        BufHost bufHost,
        Vec extents
    ) -> void
    {
        alpaka::mem::view::copy(
            devQueue,
            bufHost,
            bufAcc,
            extents
        );
    }


    auto sliceOnDevice(
        BufDevice bufferToBeSliced,
        Vec subViewExtents,
        Vec offsets
    ) -> BufDevice
    {
        BufDevice slicedBuffer = createDeviceBuffer(subViewExtents);
        // Create a subView with a possible offset.
        SubView subView = SubView(
            bufferToBeSliced,
            subViewExtents,
            offsets
        );
        // Copy the subView into a new buffer.
        alpaka::mem::view::copy(
            devQueue,
            slicedBuffer,
            subView,
            subViewExtents
        );
        return slicedBuffer;
    }


    auto compareBuffer(
        BufHost const & bufferA,
        BufHost const & bufferB,
        Vec const & extents
    ) const
    {
        TData const *const ptrA = alpaka::mem::view::getPtrNative(bufferA);
        TData const *const ptrB = alpaka::mem::view::getPtrNative(bufferB);
        for(TIdx i(0);i < extents.prod();++i)
        {
            INFO("Dim: " << TDim::value)
            INFO("Idx: " << typeid(TIdx).name())
            INFO("Acc: " << alpaka::acc::traits::GetAccName<TAcc>::getAccName())
            INFO("i: " << i)
            REQUIRE(ptrA[i] == Approx(ptrB[i]));
        }

    }
};

using DataTypes = std::tuple<
    int,
    float,
    double
>;

using TestAccWithDataTypes =
alpaka::meta::CartesianProduct<
    std::tuple,
    alpaka::test::acc::TestAccs,
    DataTypes
>;

TEMPLATE_LIST_TEST_CASE("memBufSlicingTest",
    "[memBuf]",
    TestAccWithDataTypes)
{
    using Acc = std::tuple_element_t<
        0,
        TestType
    >;
    using Data = std::tuple_element_t<
        1,
        TestType
    >;
    using Dim = alpaka::dim::Dim<Acc>;
    // fourth-dimension is not supposed to be tested currently
    if(Dim::value == 4)
    {
        return;
    }
    using Idx = alpaka::idx::Idx<Acc>;
    TestContainer<
        Dim,
        Idx,
        Acc,
        Data
    > slicingTest;

    auto const extents(
        alpaka::vec::createVecFromIndexedFn<
            Dim,
            alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf
        >());

    auto const extentsSubView(
        alpaka::vec::createVecFromIndexedFn<
            Dim,
            alpaka::test::CreateVecWithIdx<Idx>::template ForExtentSubView
        >());
    auto const offsets(
        alpaka::vec::createVecFromIndexedFn<
            Dim,
            alpaka::test::CreateVecWithIdx<Idx>::template ForOffset
        >());

    // This is the initial buffer.
    auto const indexedBuffer = slicingTest.createHostBuffer(
        extents,
        true
    );
    // This buffer will hold the sliced-buffer when it was copied to the host.
    auto resultBuffer = slicingTest.createHostBuffer(
        extentsSubView,
        false
    );

    // Copy of the indexBuffer on the deviceSide.
    auto deviceBuffer = slicingTest.createDeviceBuffer(extents);

    // Start: Main-Test
    slicingTest.copyToAcc(
        indexedBuffer,
        deviceBuffer,
        extents
    );

    auto slicedBuffer = slicingTest.sliceOnDevice(
        deviceBuffer,
        extentsSubView,
        offsets
    );

    slicingTest.copyToHost(
        slicedBuffer,
        resultBuffer,
        extentsSubView
    );

    auto correctResults = slicingTest.createHostBuffer(
        extentsSubView,
        false
    );
    Data *ptrNative = alpaka::mem::view::getPtrNative(correctResults);
    using Dim1 = alpaka::dim::DimInt<1u>;

    for(Idx i(0);i < extentsSubView.prod();++i)
    {
        auto mappedToND = alpaka::idx::mapIdx<
            Dim::value,
            Dim1::value
        >(
            alpaka::vec::Vec<
                Dim1,
                Idx
            >(i),
            extentsSubView
        );
        auto addedOffset = mappedToND + offsets;
        auto mappedTo1D = alpaka::idx::mapIdx<Dim1::value>(
            addedOffset,
            extents
        )[0]; // take the only element in the vector
        ptrNative[i] = static_cast<Data>(mappedTo1D);
    }

    // resultBuffer will be compared with the manually computed results.
    slicingTest.compareBuffer(
        resultBuffer,
        correctResults,
        extentsSubView
    );
}

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
    #pragma warning(pop)
#endif
