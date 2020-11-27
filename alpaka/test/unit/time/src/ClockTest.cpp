/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/time/Traits.hpp>

#include <catch2/catch.hpp>

//#############################################################################
class ClockTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        std::uint64_t const start(alpaka::clock(acc));
        ALPAKA_CHECK(*success, 0u != start);

        std::uint64_t const end(alpaka::clock(acc));
        ALPAKA_CHECK(*success, 0u != end);

        // 'end' has to be greater equal 'start'.
        // CUDA clock will never be equal for two calls, but the clock implementations for CPUs can be.
        ALPAKA_CHECK(*success, end >= start);
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("clockIsWorking", "[timeClock]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    ClockTestKernel kernel;

    REQUIRE(fixture(kernel));
}
