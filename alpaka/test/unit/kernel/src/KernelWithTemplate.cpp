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

#include <type_traits>

//#############################################################################
template<
    typename T>
class KernelFuntionObjectTemplate
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
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same<std::int32_t, T>::value,
            "Incorrect additional kernel template parameter type!");
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "kernelFuntionObjectTemplate", "[kernel]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelFuntionObjectTemplate<std::int32_t> kernel;

    REQUIRE(fixture(kernel));
}

//#############################################################################
class KernelInvocationWithAdditionalTemplate
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename T>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success,
        T const &) const
    -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::idx::Idx<TAcc>>(1) == (alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same<std::int32_t, T>::value,
            "Incorrect additional kernel template parameter type!");
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "kernelFuntionObjectExtraTemplate", "[kernel]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelInvocationWithAdditionalTemplate kernel;

    REQUIRE(fixture(kernel, std::int32_t()));
}
