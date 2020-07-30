/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/kernel/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/meta/ForEachType.hpp>

#include <catch2/catch.hpp>

//#############################################################################
class KernelWithAdditionalParamByValue
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        std::int32_t val) const
    -> void
    {
        alpaka::ignore_unused(acc);

        ALPAKA_CHECK(*success, 42 == val);
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("KernelWithAdditionalParamByValue", "[kernel]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelWithAdditionalParamByValue kernel;

    REQUIRE(fixture(kernel, 42));
}

/*
Passing a parameter by reference to non-const is not allowed.
There is only one single copy of the parameters on the CPU accelerators.
They are shared between all threads. Therefore they should not be mutated.

//#############################################################################
class KernelWithAdditionalParamByRef
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template <typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const &acc,
        bool *success,
        std::int32_t &val) const -> void {
        alpaka::ignore_unused(acc);

        ALPAKA_CHECK(*success, 42 == val);
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("KernelWithAdditionalParamByRef", "[kernel]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelWithAdditionalParamByRef kernel;

    REQUIRE(fixture(kernel, 42));
}*/

//#############################################################################
class KernelWithAdditionalParamByConstRef
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template <typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const &acc,
        bool *success,
        std::int32_t const &val) const -> void
    {
        alpaka::ignore_unused(acc);

        ALPAKA_CHECK(*success, 42 == val);
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("KernelWithAdditionalParamByConstRef", "[kernel]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelWithAdditionalParamByConstRef kernel;

    REQUIRE(fixture(kernel, 42));
}
