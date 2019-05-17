/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>


//#############################################################################
class ClockTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        std::uint64_t const start(
            alpaka::time::clock(acc));
        ALPAKA_CHECK(*success, 0u != start);

        std::uint64_t const end(
            alpaka::time::clock(acc));
        ALPAKA_CHECK(*success, 0u != end);

        // 'end' has to be greater equal 'start'.
        // CUDA clock will never be equal for two calls, but the clock implementations for CPUs can be.
        ALPAKA_CHECK(*success, end >= start);
    }
};

//-----------------------------------------------------------------------------
struct TestTemplate
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    ClockTestKernel kernel;

    REQUIRE(fixture(kernel));
}
};

TEST_CASE( "clockIsWorking", "[timeClock]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}
