/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <numeric>
#include <type_traits>

//-----------------------------------------------------------------------------
template<typename TAcc>
static auto testP2P(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = std::uint32_t;
    using Idx = alpaka::Idx<TAcc>;

    if(alpaka::getDevCount<Pltf>() < 2)
    {
        std::cerr << "No two devices found to test peer-to-peer copy." << std::endl;
        CHECK(true);
        return;
    }

    Dev const dev0(alpaka::getDevByIdx<Pltf>(0u));
    Dev const dev1(alpaka::getDevByIdx<Pltf>(1u));
    Queue queue0(dev0);

    //-----------------------------------------------------------------------------
    auto buf0(alpaka::allocBuf<Elem, Idx>(dev0, extent));
    auto buf1(alpaka::allocBuf<Elem, Idx>(dev1, extent));

    //-----------------------------------------------------------------------------
    std::uint8_t const byte(static_cast<uint8_t>(42u));
    alpaka::memset(queue0, buf0, byte, extent);

    //-----------------------------------------------------------------------------
    alpaka::memcpy(queue0, buf1, buf0, extent);
    alpaka::wait(queue0);
    alpaka::test::verifyBytesSet<TAcc>(buf1, byte);
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("memP2PTest", "[memP2P]", alpaka::test::TestAccs)
{
#if defined(ALPAKA_CI) && BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(7, 2, 0)                                            \
    && BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(8, 0, 0) && defined(ALPAKA_ACC_ANY_BT_OMP5_ENABLED)
    std::cerr << "Currently, memP2P is not working with gcc7.2 / gcc7.3 on CI." << std::endl;
    CHECK(true);
#else
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const extent(
        alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());

    testP2P<Acc>(extent);
#endif
}
