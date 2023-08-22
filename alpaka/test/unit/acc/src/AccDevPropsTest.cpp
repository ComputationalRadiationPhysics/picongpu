/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("getAccDevProps", "[acc]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto const devProps = alpaka::getAccDevProps<Acc>(dev);

    REQUIRE(devProps.m_gridBlockExtentMax.min() > 0);
    REQUIRE(devProps.m_blockThreadExtentMax.min() > 0);
    REQUIRE(devProps.m_threadElemExtentMax.min() > 0);
    REQUIRE(devProps.m_gridBlockCountMax > 0);
    REQUIRE(devProps.m_blockThreadCountMax > 0);
    REQUIRE(devProps.m_threadElemCountMax > 0);
    REQUIRE(devProps.m_multiProcessorCount > 0);
    REQUIRE(devProps.m_sharedMemSizeBytes > 0);
}
