/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

template<typename TAcc>
static auto testP2P(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = std::uint32_t;
    using Idx = alpaka::Idx<TAcc>;

    auto const platformAcc = alpaka::Platform<TAcc>{};
    if(alpaka::getDevCount(platformAcc) < 2)
    {
        std::cerr << "No two devices found to test peer-to-peer copy." << std::endl;
        CHECK(true);
        return;
    }

    Dev const dev0 = alpaka::getDevByIdx(platformAcc, 0);
    Dev const dev1 = alpaka::getDevByIdx(platformAcc, 1);
    Queue queue0(dev0);
    Queue queue1(dev1);

    auto buf0 = alpaka::allocBuf<Elem, Idx>(dev0, extent);
    auto buf1 = alpaka::allocBuf<Elem, Idx>(dev1, extent);

    // fill each byte with value 42
    auto const byte(static_cast<uint8_t>(42u));
    alpaka::memset(queue1, buf1, byte);
    alpaka::wait(queue1);

    // copy buffer from device 1 into device 0 buffer
    alpaka::memcpy(queue0, buf0, buf1);
    alpaka::wait(queue0);
    // verify buffer on device 0
    alpaka::test::verifyBytesSet<TAcc>(buf0, byte);
}

TEMPLATE_LIST_TEST_CASE("memP2PTest", "[memP2P]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    testP2P<Acc>(alpaka::test::extentBuf<Dim, Idx>);
}
