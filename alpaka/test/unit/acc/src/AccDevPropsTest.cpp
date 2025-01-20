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
    REQUIRE(devProps.m_globalMemSizeBytes > 0);
}

TEST_CASE("AccDevProps.aggregate_init", "[acc]")
{
    auto const props = alpaka::AccDevProps<alpaka::DimInt<1>, int>{1, {2}, 3, {4}, 5, {6}, 7, 8, 9};

    CHECK(props.m_multiProcessorCount == 1);
    CHECK(props.m_gridBlockExtentMax == alpaka::Vec{2});
    CHECK(props.m_gridBlockCountMax == 3);
    CHECK(props.m_blockThreadExtentMax == alpaka::Vec{4});
    CHECK(props.m_blockThreadCountMax == 5);
    CHECK(props.m_threadElemExtentMax == alpaka::Vec{6});
    CHECK(props.m_threadElemCountMax == 7);
    CHECK(props.m_sharedMemSizeBytes == 8);
    CHECK(props.m_globalMemSizeBytes == 9);
}

#ifdef __cpp_designated_initializers
TEST_CASE("AccDevProps.designated_initializers", "[acc]")
{
    auto const props = alpaka::AccDevProps<alpaka::DimInt<1>, int>{
        .m_multiProcessorCount = 10,
        .m_gridBlockExtentMax = {20},
        .m_gridBlockCountMax = 30,
        .m_blockThreadExtentMax = {40},
        .m_blockThreadCountMax = 50,
        .m_threadElemExtentMax = {60},
        .m_threadElemCountMax = 70,
        .m_sharedMemSizeBytes = 80,
        .m_globalMemSizeBytes = 90};

    CHECK(props.m_multiProcessorCount == 10);
    CHECK(props.m_gridBlockExtentMax == alpaka::Vec{20});
    CHECK(props.m_gridBlockCountMax == 30);
    CHECK(props.m_blockThreadExtentMax == alpaka::Vec{40});
    CHECK(props.m_blockThreadCountMax == 50);
    CHECK(props.m_threadElemExtentMax == alpaka::Vec{60});
    CHECK(props.m_threadElemCountMax == 70);
    CHECK(props.m_sharedMemSizeBytes == 80);
    CHECK(props.m_globalMemSizeBytes == 90);
}
#endif
