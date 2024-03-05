/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/kernel/Traits.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

class KernelWithHostConstexpr
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& /* acc */, bool* success) const -> void
    {
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(push)
#    pragma warning(disable : 4127) // warning C4127: conditional expression is constant
#endif

        constexpr auto max = std::numeric_limits<std::uint32_t>::max();

        ALPAKA_CHECK(*success, 0 != max);
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif
    }
};

TEMPLATE_LIST_TEST_CASE("kernelWithHostConstexpr", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    KernelWithHostConstexpr kernel;

    REQUIRE(fixture(kernel));
}
