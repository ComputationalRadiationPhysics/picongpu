/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("isSingleThreadAcc", "[acc]", alpaka::test::TestAccs)
{
    using Acc = TestType;

    // Check that both traits are defined, and that only one is true.
    REQUIRE(alpaka::isSingleThreadAcc<Acc> != alpaka::isMultiThreadAcc<Acc>);

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto const devProps = alpaka::getAccDevProps<Acc>(dev);

    // Compare the runtime properties with the compile time trait.
    INFO("Accelerator: " << alpaka::core::demangled<Acc>);
    if constexpr(alpaka::isSingleThreadAcc<Acc>)
    {
        // Require a single thread per block.
        REQUIRE(devProps.m_blockThreadCountMax == 1);
    }
    else
    {
        // Assume multiple threads per block, but allow a single thread per block.
        // For example, the AccCpuOmp2Threads accelerator may report a single thread on a single core system.
        REQUIRE(devProps.m_blockThreadCountMax >= 1);
    }
}
