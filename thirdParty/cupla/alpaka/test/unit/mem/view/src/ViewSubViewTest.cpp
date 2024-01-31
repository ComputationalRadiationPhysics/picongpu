/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'std::uint8_t*' to 'Elem*' increases required alignment of target type"
#endif

namespace alpaka::test
{
    template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx, typename TBuf>
    auto testViewSubViewImmutable(
        alpaka::ViewSubView<TDev, TElem, TDim, TIdx> const& view,
        TBuf& buf,
        TDev const& dev,
        alpaka::Vec<TDim, TIdx> const& extentView,
        alpaka::Vec<TDim, TIdx> const& offsetView) -> void
    {
        alpaka::test::testViewImmutable<TElem>(view, dev, extentView, offsetView);

        // alpaka::trait::GetPitchesInBytes
        // The pitch of the view has to be identical to the pitch of the underlying buffer in all dimensions.
        auto const pitchBuf = alpaka::getPitchesInBytes(buf);
        {
            auto const pitchView = alpaka::getPitchesInBytes(view);
            CHECK(pitchBuf == pitchView);
        }

        // alpaka::trait::GetPtrNative
        // The native pointer has to be exactly the value we calculate here.
        {
            auto viewPtrNative = reinterpret_cast<std::uint8_t*>(alpaka::getPtrNative(buf));
            if constexpr(TDim::value > 0)
                viewPtrNative += (offsetView * pitchBuf).sum();
            REQUIRE(reinterpret_cast<TElem*>(viewPtrNative) == alpaka::getPtrNative(view));
        }
    }

    template<typename TAcc, typename TDev, typename TElem, typename TDim, typename TIdx, typename TBuf>
    auto testViewSubViewMutable(
        alpaka::ViewSubView<TDev, TElem, TDim, TIdx>& view,
        TBuf& buf,
        TDev const& dev,
        alpaka::Vec<TDim, TIdx> const& extentView,
        alpaka::Vec<TDim, TIdx> const& offsetView) -> void
    {
        testViewSubViewImmutable<TAcc>(view, buf, dev, extentView, offsetView);

        using Queue = alpaka::test::DefaultQueue<TDev>;
        Queue queue(dev);
        alpaka::test::testViewMutable<TAcc>(queue, view);
    }

    template<typename TAcc, typename TElem>
    auto testViewSubViewNoOffset() -> void
    {
        using Dev = alpaka::Dev<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using View = alpaka::ViewSubView<Dev, TElem, Dim, Idx>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);

        auto const bufExtent = alpaka::test::extentBuf<Dim, Idx>;
        auto buf = alpaka::allocBuf<TElem, Idx>(dev, bufExtent);

        auto const viewExtent = bufExtent;
        auto const viewOffset = alpaka::Vec<Dim, Idx>::zeros();
        View view(buf);

        alpaka::test::testViewSubViewMutable<TAcc>(view, buf, dev, viewExtent, viewOffset);
    }

    template<typename TAcc, typename TElem>
    auto testViewSubViewOffset() -> void
    {
        using Dev = alpaka::Dev<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using View = alpaka::ViewSubView<Dev, TElem, Dim, Idx>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);

        auto const bufExtent = alpaka::test::extentBuf<Dim, Idx>;
        auto buf = alpaka::allocBuf<TElem, Idx>(dev, bufExtent);

        auto const viewExtent = alpaka::test::extentSubView<Dim, Idx>;
        auto const viewOffset = alpaka::test::offset<Dim, Idx>;
        View view(buf, viewExtent, viewOffset);

        alpaka::test::testViewSubViewMutable<TAcc>(view, buf, dev, viewExtent, viewOffset);
    }

    template<typename TAcc, typename TElem>
    auto testViewSubViewOffsetConst() -> void
    {
        using Dev = alpaka::Dev<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using View = alpaka::ViewSubView<Dev, TElem, Dim, Idx>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);

        auto const bufExtent = alpaka::test::extentBuf<Dim, Idx>;
        auto buf = alpaka::allocBuf<TElem, Idx>(dev, bufExtent);

        auto const viewExtent = alpaka::test::extentSubView<Dim, Idx>;
        auto const viewOffset = alpaka::test::offset<Dim, Idx>;
        View const view(buf, viewExtent, viewOffset);

        alpaka::test::testViewSubViewImmutable<TAcc>(view, buf, dev, viewExtent, viewOffset);
    }
} // namespace alpaka::test
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

TEMPLATE_LIST_TEST_CASE("viewSubViewNoOffsetTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::testViewSubViewNoOffset<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewSubViewOffsetTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::testViewSubViewOffset<TestType, float>();
}

TEMPLATE_LIST_TEST_CASE("viewSubViewOffsetConstTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::testViewSubViewOffsetConst<TestType, float>();
}

TEST_CASE("viewSubViewExample", "[memView]")
{
    using Dev = alpaka::DevCpu;
    using Dim = alpaka::DimInt<2>;
    using Idx = int;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const platform = alpaka::PlatformCpu{};
    auto const dev = alpaka::getDevByIdx(platform, 0);

    auto const extent = Vec{4, 5};
    auto buf = alpaka::allocBuf<int, Idx>(dev, extent);

    auto checkBufContent = [&](std::vector<std::vector<int>> const& data)
    {
        for(std::size_t row = 0; row < 4; row++)
            for(std::size_t col = 0; col < 5; col++)
            {
                CAPTURE(row, col);
                CHECK(buf[Vec{static_cast<Idx>(row), static_cast<Idx>(col)}] == data[row][col]);
            }
    };

    for(Idx i = 0; i < 4; i++)
        for(Idx j = 0; j < 5; j++)
            buf.at(Vec{i, j}) = 1;
    checkBufContent({{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}});

    {
        // write a row
        auto subView = alpaka::ViewSubView<Dev, int, Dim, Idx>{buf, Vec{1, 5}, Vec{2, 0}};
        for(Idx i = 0; i < 5; i++)
            subView.at(Vec{0, i}) = 2;
    }
    checkBufContent({{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {1, 1, 1, 1, 1}});

    {
        // write part of a column
        auto subView = alpaka::ViewSubView<Dev, int, Dim, Idx>{buf, Vec{3, 1}, Vec{0, 2}};
        for(Idx i = 0; i < 3; i++)
            subView.at(Vec{i, 0}) = 3;
    }
    checkBufContent({{1, 1, 3, 1, 1}, {1, 1, 3, 1, 1}, {2, 2, 3, 2, 2}, {1, 1, 1, 1, 1}});

    {
        // write the center without a 1-elem border
        auto subView = alpaka::ViewSubView<Dev, int, Dim, Idx>{buf, Vec{2, 3}, Vec{1, 1}};
        for(Idx i = 0; i < 2; i++)
            for(Idx j = 0; j < 3; j++)
                subView.at(Vec{i, j}) = 4;
        checkBufContent({{1, 1, 3, 1, 1}, {1, 4, 4, 4, 1}, {2, 4, 4, 4, 2}, {1, 1, 1, 1, 1}});

        {
            // write the first column of the center region
            auto subSubView = alpaka::ViewSubView<Dev, int, Dim, Idx>{subView, Vec{2, 1}, Vec{0, 0}};
            for(Idx i = 0; i < 2; i++)
                subSubView.at(Vec{i, 0}) = 5;
        }
        checkBufContent({{1, 1, 3, 1, 1}, {1, 5, 4, 4, 1}, {2, 5, 4, 4, 2}, {1, 1, 1, 1, 1}});

        {
            // write the second row of the center region, skipping the first
            auto subSubView = alpaka::ViewSubView<Dev, int, Dim, Idx>{subView, Vec{1, 2}, Vec{1, 1}};
            for(Idx i = 0; i < 2; i++)
                subSubView.at(Vec{0, i}) = 6;
        }
        checkBufContent({{1, 1, 3, 1, 1}, {1, 5, 4, 4, 1}, {2, 5, 6, 6, 2}, {1, 1, 1, 1, 1}});
    }
}

TEST_CASE("calculatePitchesFromExtents", "[memView]")
{
    CHECK((alpaka::detail::calculatePitchesFromExtents<float>(alpaka::Vec{1, 1, 1}) == alpaka::Vec{4, 4, 4}));
    CHECK((alpaka::detail::calculatePitchesFromExtents<float>(alpaka::Vec{2, 2, 2}) == alpaka::Vec{16, 8, 4}));
    CHECK((alpaka::detail::calculatePitchesFromExtents<float>(alpaka::Vec{42, 10, 2}) == alpaka::Vec{80, 8, 4}));
}
