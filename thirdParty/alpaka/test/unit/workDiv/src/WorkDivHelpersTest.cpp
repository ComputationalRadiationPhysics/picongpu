/* Copyright 2022 Sergei Bastrakov, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tuple>

namespace
{
    template<typename TAcc>
    auto getWorkDiv()
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);
        auto const gridThreadExtent = alpaka::Vec<Dim, Idx>::all(10);
        auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
        auto const workDiv = alpaka::getValidWorkDiv<TAcc>(
            dev,
            gridThreadExtent,
            threadElementExtent,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);
        return workDiv;
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("getValidWorkDiv", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    // Note: getValidWorkDiv() is called inside getWorkDiv
    std::ignore = getWorkDiv<Acc>();
}

TEMPLATE_LIST_TEST_CASE("subDivideGridElems.2D.examples", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    if constexpr(Dim::value == 2)
    {
        auto const platform = alpaka::Platform<Acc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);
        auto props = alpaka::getAccDevProps<Acc>(dev);
        props.m_gridBlockExtentMax = Vec{1024, 1024};
        props.m_gridBlockCountMax = 1024 * 1024;
        props.m_blockThreadExtentMax = Vec{256, 128};
        props.m_blockThreadCountMax = 512;
        props.m_threadElemExtentMax = Vec{8, 8};
        props.m_threadElemCountMax = 16;

        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                false,
                alpaka::GridBlockExtentSubDivRestrictions::EqualExtent)
            == WorkDiv{Vec{14, 28}, Vec{22, 22}, Vec{1, 1}});
        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                false,
                alpaka::GridBlockExtentSubDivRestrictions::CloseToEqualExtent)
            == WorkDiv{Vec{19, 19}, Vec{16, 32}, Vec{1, 1}});
        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                false,
                alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)
            == WorkDiv{Vec{75, 5}, Vec{4, 128}, Vec{1, 1}});

        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                true,
                alpaka::GridBlockExtentSubDivRestrictions::EqualExtent)
            == WorkDiv{Vec{1, 2}, Vec{300, 300}, Vec{1, 1}});
        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                true,
                alpaka::GridBlockExtentSubDivRestrictions::CloseToEqualExtent)
            == WorkDiv{Vec{20, 20}, Vec{15, 30}, Vec{1, 1}});
        CHECK(
            alpaka::subDivideGridElems(
                Vec{300, 600},
                Vec{1, 1},
                props,
                true,
                alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)
            == WorkDiv{Vec{75, 5}, Vec{4, 120}, Vec{1, 1}});
    }
}

TEMPLATE_LIST_TEST_CASE("getValidWorkDiv.1D.withIdx", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    if constexpr(Dim::value == 1)
    {
        auto const platform = alpaka::Platform<Acc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);
        // test that we can call getValidWorkDiv with the Idx type directly instead of a Vec
        auto const ref = alpaka::getValidWorkDiv<Acc>(dev, Vec{256}, Vec{13});
        CHECK(alpaka::getValidWorkDiv<Acc>(dev, Idx{256}, Idx{13}) == ref);
    }
}

TEMPLATE_LIST_TEST_CASE("isValidWorkDiv", "[workDiv]", alpaka::test::TestAccs)
{
    using Acc = TestType;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto const workDiv = getWorkDiv<Acc>();
    // Test both overloads
    REQUIRE(alpaka::isValidWorkDiv(alpaka::getAccDevProps<Acc>(dev), workDiv));
    REQUIRE(alpaka::isValidWorkDiv<Acc>(dev, workDiv));
}
