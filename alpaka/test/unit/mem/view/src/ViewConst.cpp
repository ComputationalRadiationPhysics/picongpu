/* Copyright 2022 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/view/ViewConst.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <numeric>
#include <type_traits>

namespace alpaka::test
{
    template<typename TAcc, typename TQualifiedView, typename TDev, typename TQueue, typename TDim, typename TIdx>
    void testViewConst(
        TQualifiedView& view,
        TDev const& dev,
        TQueue& queue,
        alpaka::Vec<TDim, TIdx> const& extents,
        alpaka::Vec<TDim, TIdx> const& offsets)
    {
        using TView = std::remove_const_t<TQualifiedView>;
        STATIC_REQUIRE(std::is_same_v<alpaka::Elem<TView>, float const>);
        STATIC_REQUIRE(std::is_same_v<decltype(alpaka::getPtrNative(view)), float const*>);

        alpaka::test::testViewImmutable<float const>(view, dev, extents, offsets);

        // test copying from view
        auto dstBuf = allocBuf<float, TIdx>(dev, getExtents(view));
        memcpy(queue, dstBuf, view);
        wait(queue);
        verifyViewsEqual<TAcc>(dstBuf, view);

        // test special member functions
        TView viewCopy(view);
        TView viewMove(std::move(viewCopy));
        viewCopy = viewMove;
        viewCopy = std::move(viewMove);

        // test view accessors
        STATIC_REQUIRE(std::is_same_v<decltype(view.data()), float const*>);
        if constexpr(TDim::value == 0)
        {
            STATIC_REQUIRE(std::is_same_v<decltype(*view), float const&>);
        }
        else if constexpr(TDim::value == 1)
        {
            STATIC_REQUIRE(std::is_same_v<decltype(view[0]), float const&>);
        }

        STATIC_REQUIRE(std::is_same_v<decltype(view[alpaka::Vec<TDim, TIdx>::zeros()]), float const&>);
        STATIC_REQUIRE(std::is_same_v<decltype(view.at(alpaka::Vec<TDim, TIdx>::zeros())), float const&>);
    }
} // namespace alpaka::test

TEMPLATE_LIST_TEST_CASE("viewConstTest", "[memView]", alpaka::test::TestAccs)
{
    using Elem = float;
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    auto const extents = alpaka::test::extentBuf<Dim, Idx>;
    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extents);
    auto queue = alpaka::test::DefaultQueue<Dev>{dev};
    alpaka::test::iotaFillView(queue, buf);
    auto const offsets = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(0));

    auto view = alpaka::ViewConst(buf);
    alpaka::test::testViewConst<Acc>(view, dev, queue, extents, offsets);

    auto const cview = alpaka::ViewConst(buf);
    alpaka::test::testViewConst<Acc>(cview, dev, queue, extents, offsets);

    auto view_cbuf = alpaka::ViewConst(std::as_const(buf));
    alpaka::test::testViewConst<Acc>(view_cbuf, dev, queue, extents, offsets);

    auto const cview_cbuf = alpaka::ViewConst(std::as_const(buf));
    alpaka::test::testViewConst<Acc>(cview_cbuf, dev, queue, extents, offsets);

    auto yolo = alpaka::ViewConst(alpaka::ViewConst(alpaka::ViewConst(cview_cbuf)));
    alpaka::test::testViewConst<Acc>(yolo, dev, queue, extents, offsets);
}
