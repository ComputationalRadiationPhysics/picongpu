/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
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
class KernelWithConstructorAndMember
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_FN_HOST KernelWithConstructorAndMember(
        std::int32_t const val = 42) :
        m_val(val)
    {}

    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        alpaka::ignore_unused(acc);

        ALPAKA_CHECK(*success, 42 == m_val);
    }

private:
    std::int32_t m_val;
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

        KernelWithConstructorAndMember kernel(42);

        REQUIRE(fixture(kernel));
    }
};

//-----------------------------------------------------------------------------
struct TestTemplateDefault
{
    template< typename TAcc >
    void operator()()
    {
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(
            alpaka::vec::Vec<Dim, Idx>::ones());

        KernelWithConstructorAndMember kernel;

        REQUIRE(fixture(kernel));
    }
};

TEST_CASE( "kernelWithConstructorAndMember", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}

TEST_CASE( "kernelWithConstructorDefaultParamAndMember", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateDefault() );
}
