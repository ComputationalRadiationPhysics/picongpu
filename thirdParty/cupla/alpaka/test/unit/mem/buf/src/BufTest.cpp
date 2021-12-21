/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <numeric>
#include <type_traits>

//-----------------------------------------------------------------------------
template<typename TAcc>
static auto testBufferMutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
    Queue queue(dev);

    //-----------------------------------------------------------------------------
    // alpaka::malloc
    auto buf(alpaka::allocBuf<Elem, Idx>(dev, extent));

    //-----------------------------------------------------------------------------
    auto const offset(alpaka::Vec<Dim, Idx>::zeros());
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);

    //-----------------------------------------------------------------------------
    alpaka::test::testViewMutable<TAcc>(queue, buf);
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("memBufBasicTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const extent(
        alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());

    testBufferMutable<Acc>(extent);
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("memBufZeroSizeTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const extent(alpaka::Vec<Dim, Idx>::zeros());

    testBufferMutable<Acc>(extent);
}


//-----------------------------------------------------------------------------
template<typename TAcc>
static auto testBufferImmutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Pltf = alpaka::Pltf<Dev>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));

    //-----------------------------------------------------------------------------
    // alpaka::malloc
    auto const buf(alpaka::allocBuf<Elem, Idx>(dev, extent));

    //-----------------------------------------------------------------------------
    auto const offset(alpaka::Vec<Dim, Idx>::zeros());
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("memBufConstTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const extent(
        alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());

    testBufferImmutable<Acc>(extent);
}
