/* Copyright 2026 Axel Hübl, Benjamin Worpitz, Erik Zenker, Bernhard Manfred Gruber, Jan Stephan, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/Config.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <algorithm>
#include <numeric>
#include <tuple>
#include <type_traits>

namespace alpaka::test
{
    template<typename TAcc, typename TElem, bool Const>
    auto testViewPlainPtr() -> void
    {
        using Dev = alpaka::Dev<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using View = alpaka::ViewPlainPtr<Dev, TElem, Dim, Idx>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);

        auto const bufExtent = alpaka::test::extentBuf<Dim, Idx>;
        auto buf = alpaka::allocBuf<TElem, Idx>(dev, bufExtent);

        auto const viewExtent = bufExtent;
        auto const viewOffset = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(0));
        View view(
            alpaka::getPtrNative(buf),
            alpaka::getDev(buf),
            alpaka::getExtents(buf),
            alpaka::getPitchesInBytes(buf));

        alpaka::test::testViewImmutable<TElem>(std::as_const(view), dev, viewExtent, viewOffset);
        if constexpr(!Const)
        {
            using Queue = alpaka::test::DefaultQueue<Dev>;
            Queue queue(dev);
            alpaka::test::testViewMutable<TAcc>(queue, view);
        }
    }

    template<typename TAcc, typename TElem>
    auto testViewPlainPtrOperators() -> void
    {
        using Dev = alpaka::Dev<TAcc>;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;
        using View = alpaka::ViewPlainPtr<Dev, TElem, Dim, Idx>;

        auto const platform = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platform, 0);

        auto buf = alpaka::allocBuf<TElem, Idx>(dev, alpaka::test::extentBuf<Dim, Idx>);
        auto nativePtr = alpaka::getPtrNative(buf);
        View view(nativePtr, alpaka::getDev(buf), alpaka::getExtents(buf), alpaka::getPitchesInBytes(buf));

        // copy-constructor
        View viewCopy(view);
        CHECK(alpaka::getPtrNative(viewCopy) == nativePtr);

        // move-constructor
        View viewMove(std::move(viewCopy));
        CHECK(alpaka::getPtrNative(viewMove) == nativePtr);

        auto buf2 = alpaka::allocBuf<TElem, Idx>(dev, alpaka::test::extentBuf<Dim, Idx>);
        auto nativePtr2 = alpaka::getPtrNative(buf);
        View view2(nativePtr2, alpaka::getDev(buf2), alpaka::getExtents(buf2), alpaka::getPitchesInBytes(buf2));

        // copy-assign
        viewCopy = view2;
        CHECK(alpaka::getPtrNative(viewCopy) == nativePtr2);

        // move-assign
        viewMove = std::move(view2);
        CHECK(alpaka::getPtrNative(viewMove) == nativePtr2);
    }
} // namespace alpaka::test

TEMPLATE_LIST_TEST_CASE("viewPlainPtrTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::testViewPlainPtr<TestType, float, false>();
}

TEMPLATE_LIST_TEST_CASE("viewPlainPtrConstTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::testViewPlainPtr<TestType, float, true>();
}

TEMPLATE_LIST_TEST_CASE("viewPlainPtrOperatorTest", "[memView]", alpaka::test::TestAccs)
{
    alpaka::test::testViewPlainPtrOperators<TestType, float>();
}

// TODO(SimeonEhrig): if C++ 20 is minimum requirement, remove `using`
// replace it with `TEMPLATE_TEST_CASE("Vec generator constructor", "[vec]", std::size_t, int, unsigned, float,
// double)`
// missing feature: parameter of a variadic macro is a C++20 extension
using CreateViewTestTypes = std::tuple<std::array<float, 4>, std::vector<float>>;

TEMPLATE_LIST_TEST_CASE("createView", "[memView]", CreateViewTestTypes)
{
    using Dev = alpaka::DevCpu;
    auto const platform = alpaka::PlatformCpu{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    alpaka::test::DefaultQueue<Dev> queue(dev);

    TestType a{1, 2, 3, 4};

    // pointer overload
    {
        auto view = alpaka::createView(dev, a.data(), 4);
        STATIC_REQUIRE(std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, float, alpaka::DimInt<1>, int>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);

        auto view_from_queue = alpaka::createView(queue, a.data(), 4);
        STATIC_REQUIRE(std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, float, alpaka::DimInt<1>, int>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);
    }

    // container and size overload
    {
        auto view = alpaka::createView(dev, a, 4L);
        STATIC_REQUIRE(std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, float, alpaka::DimInt<1>, long>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);

        auto view_from_queue = alpaka::createView(queue, a, 4L);
        STATIC_REQUIRE(std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, float, alpaka::DimInt<1>, long>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);
    }

    // container overload
    {
        auto view = alpaka::createView(dev, a);
        STATIC_REQUIRE(
            std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, float, alpaka::DimInt<1>, std::size_t>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);

        auto view_from_queue = alpaka::createView(queue, a);
        STATIC_REQUIRE(
            std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, float, alpaka::DimInt<1>, std::size_t>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);
    }

    decltype(a) b{0, 0, 0, 0};

    // test as temporaries to memcpy
    alpaka::memcpy(queue, alpaka::createView(queue, b, std::size_t{4}), alpaka::createView(queue, a));
    alpaka::wait(queue);
    CHECK(a == b);

    // test as temporaries to memset
    alpaka::memset(queue, alpaka::createView(queue, a), 0);
    alpaka::wait(queue);
    CHECK(a == decltype(a){0, 0, 0, 0});
}

TEMPLATE_LIST_TEST_CASE("createViewSpan", "[memView]", std::tuple<std::span<int>>)
{
    using Dev = alpaka::DevCpu;
    auto const platform = alpaka::PlatformCpu{};
    auto const dev = alpaka::getDevByIdx(platform, 0);

    std::vector vec{1, 2, 3, 4};
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif

    std::span<int> a(vec.begin(), vec.end());
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif


    // pointer overload
    {
        auto view = alpaka::createView(dev, a.data(), 4);
        STATIC_REQUIRE(std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, int, alpaka::DimInt<1>, int>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);
    }

    // container and size overload
    {
        auto view = alpaka::createView(dev, a, 4L);
        STATIC_REQUIRE(std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, int, alpaka::DimInt<1>, long>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);
    }

    // container overload
    {
        auto view = alpaka::createView(dev, a);
        STATIC_REQUIRE(std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, int, alpaka::DimInt<1>, std::size_t>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);
    }

    alpaka::test::DefaultQueue<Dev> queue(dev);
    std::vector vec2{0, 0, 0, 0};
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif
    std::span<int> b(vec2.begin(), vec2.end());
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif

    // test as temporaries to memcpy
    alpaka::memcpy(queue, alpaka::createView(queue, b, std::size_t{4}), alpaka::createView(queue, a));
    alpaka::wait(queue);
    CHECK(std::equal(a.begin(), a.end(), b.begin()));

    // test as temporaries to memset
    alpaka::memset(queue, alpaka::createView(queue, a), 0);
    alpaka::wait(queue);
    for(auto x : a)
    {
        CHECK(x == 0);
    }
}

TEMPLATE_LIST_TEST_CASE("createViewConstSpan", "[memView]", std::tuple<std::span<int const>>)
{
    using Dev = alpaka::DevCpu;
    auto const platform = alpaka::PlatformCpu{};
    auto const dev = alpaka::getDevByIdx(platform, 0);

    std::vector vec{1, 2, 3, 4};
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif

    std::span<int const> a(vec.begin(), vec.end());
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif


    // pointer overload
    {
        auto view = alpaka::createView(dev, a.data(), 4);
        STATIC_REQUIRE(std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, int const, alpaka::DimInt<1>, int>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);
    }

    // container and size overload
    {
        auto view = alpaka::createView(dev, a, 4L);
        STATIC_REQUIRE(std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, int const, alpaka::DimInt<1>, long>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);
    }

    // container overload
    {
        auto view = alpaka::createView(dev, a);
        STATIC_REQUIRE(
            std::is_same_v<decltype(view), alpaka::ViewPlainPtr<Dev, int const, alpaka::DimInt<1>, std::size_t>>);
        STATIC_REQUIRE(alpaka::Dim<decltype(view)>::value == 1);
        CHECK(alpaka::getExtents(view)[0] == 4);
    }

    alpaka::test::DefaultQueue<Dev> queue(dev);
    std::vector vec2{0, 0, 0, 0};
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    if __has_warning("-Wunsafe-buffer-usage-in-container")
#        pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
#    endif
#endif
    std::span<int> b(vec2.begin(), vec2.end());
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif

    // test as temporaries to memcpy
    alpaka::memcpy(queue, alpaka::createView(queue, b, std::size_t{4}), alpaka::createView(queue, a));
    alpaka::wait(queue);
    CHECK(std::equal(a.begin(), a.end(), b.begin()));
}
