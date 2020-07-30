/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/intrinsic/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

//#############################################################################
template<
    typename TInput>
class PopcountTestKernel
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
        // Use negative values to get inputs near the max value of TInput type
        TInput const inputs[] = {0u, 1u, 3u, 54u, 163u, 51362u,
            static_cast<TInput>(-43631), static_cast<TInput>(-1352),
            static_cast<TInput>(-642), static_cast<TInput>(-1)};
        for( auto const input : inputs )
        {
            int const expected = popcountNaive(input);
            int const actual = alpaka::intrinsic::popcount(acc, input);
            ALPAKA_CHECK(*success, actual == expected);
        }
    }

private:
    ALPAKA_FN_ACC static auto popcountNaive(TInput value) -> int
    {
        int result = 0;
        while (value)
        {
            result += static_cast<int>(value & 1u);
            value >>= 1u;
        }
        return result;
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "popcount", "[intrinsic]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    PopcountTestKernel<std::uint32_t> kernel32bit;
    REQUIRE(
        fixture(
            kernel32bit));

    PopcountTestKernel<std::uint64_t> kernel64bit;
    REQUIRE(
        fixture(
            kernel64bit));
}
