/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/mem/buf/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/Extent.hpp>

#include <catch2/catch.hpp>

#include <type_traits>
#include <numeric>

//-----------------------------------------------------------------------------
template<
    typename TAcc>
static auto testBufferMutable(
    alpaka::vec::Vec<alpaka::dim::Dim<TAcc>, alpaka::idx::Idx<TAcc>> const & extent)
-> void
{
    using Dev = alpaka::dev::Dev<TAcc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;
    using Queue = alpaka::test::queue::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
    Queue queue(dev);

    //-----------------------------------------------------------------------------
    // alpaka::mem::buf::alloc
    auto buf(alpaka::mem::buf::alloc<Elem, Idx>(dev, extent));

    //-----------------------------------------------------------------------------
    auto const offset(alpaka::vec::Vec<Dim, Idx>::zeros());
    alpaka::test::mem::view::testViewImmutable<
        Elem>(
            buf,
            dev,
            extent,
            offset);

    //-----------------------------------------------------------------------------
    alpaka::test::mem::view::testViewMutable<
        TAcc>(
            queue,
            buf);
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "memBufBasicTest", "[memBuf]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    auto const extent(alpaka::vec::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());

    testBufferMutable<
        Acc>(
            extent);
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "memBufZeroSizeTest", "[memBuf]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    auto const extent(alpaka::vec::Vec<Dim, Idx>::zeros());

    testBufferMutable<
        Acc>(
            extent);
}


//-----------------------------------------------------------------------------
template<
    typename TAcc>
static auto testBufferImmutable(
    alpaka::vec::Vec<alpaka::dim::Dim<TAcc>, alpaka::idx::Idx<TAcc>> const & extent)
-> void
{
    using Dev = alpaka::dev::Dev<TAcc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;

    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

    //-----------------------------------------------------------------------------
    // alpaka::mem::buf::alloc
    auto const buf(alpaka::mem::buf::alloc<Elem, Idx>(dev, extent));

    //-----------------------------------------------------------------------------
    auto const offset(alpaka::vec::Vec<Dim, Idx>::zeros());
    alpaka::test::mem::view::testViewImmutable<
        Elem>(
            buf,
            dev,
            extent,
            offset);
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "memBufConstTest", "[memBuf]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    auto const extent(alpaka::vec::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());

    testBufferImmutable<
        Acc>(
            extent);
}
