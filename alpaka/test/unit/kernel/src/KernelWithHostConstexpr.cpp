/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
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

#include <limits>

//#############################################################################
class KernelWithHostConstexpr
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool* success) const
    -> void
    {
        alpaka::ignore_unused(acc);

#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
    #pragma warning(push)
    #pragma warning(disable: 4127)  // warning C4127: conditional expression is constant
#endif

        constexpr auto max = std::numeric_limits< std::uint32_t >::max();

        ALPAKA_CHECK(*success, 0 != max);
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
    #pragma warning(pop)
#endif
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "kernelWithHostConstexpr", "[kernel]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelWithHostConstexpr kernel;

    REQUIRE(fixture(kernel));
}
