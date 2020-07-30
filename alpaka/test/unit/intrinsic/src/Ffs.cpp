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

#include <cstdint>
#include <limits>

//#############################################################################
template<
    typename TInput>
class FfsTestKernel
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
        TInput const inputs[] = {0, 1, 3, 64, 256, 51362,
            std::numeric_limits<TInput>::max(),
            -1, -32, -1352, -4096, std::numeric_limits<TInput>::min()};
        for( auto const input : inputs )
        {
            std::int32_t const expected = ffsNaive(input);
            std::int32_t const actual = alpaka::intrinsic::ffs(acc, input);
            ALPAKA_CHECK(*success, actual == expected);
        }
    }

private:
    ALPAKA_FN_ACC static auto ffsNaive(TInput value) -> std::int32_t
    {
        if (value == 0)
            return 0;
        std::int32_t result = 1;
        while ((value & 1) == 0)
        {
            value >>= 1;
            result++;
        }
        return result;
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "ffs", "[intrinsic]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    FfsTestKernel<std::int32_t> kernel32bit;
    REQUIRE(
        fixture(
            kernel32bit));

    FfsTestKernel<std::int64_t> kernel64bit;
    REQUIRE(
        fixture(
            kernel64bit));
}
