/* Copyright 2022 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#ifdef ALPAKA_USE_MDSPAN
#    include <alpaka/alpaka.hpp>
#    include <alpaka/test/Extent.hpp>
#    include <alpaka/test/acc/TestAccs.hpp>
#    include <alpaka/test/mem/view/ViewTest.hpp>

#    include <catch2/catch_template_test_macros.hpp>

#    if defined(__NVCC__) && !defined(__CUDACC_EXTENDED_LAMBDA__)
#        error "mdspan requires nvcc's extended lambdas"
#    endif

namespace
{
    template<std::size_t... Is>
    constexpr auto make_reverse_index_sequence_impl(std::index_sequence<Is...>)
    {
        return std::index_sequence<(sizeof...(Is) - Is - 1)...>{};
    }

    template<std::size_t N>
    using make_reverse_index_sequence = decltype(make_reverse_index_sequence_impl(std::make_index_sequence<N>{}));
} // namespace

#    if BOOST_COMP_NVCC
#        define NOEXCEPT_UNLESS_NVCC
#    else
#        define NOEXCEPT_UNLESS_NVCC noexcept
#    endif

TEMPLATE_LIST_TEST_CASE("mdSpan", "[memView]", alpaka::test::TestAccs)
{
    using TElem = int;

    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const platform = alpaka::Platform<Dev>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Dev, alpaka::Blocking>(dev);

    auto const extent = alpaka::test::extentBuf<Dim, Idx>;
    auto buf = alpaka::allocBuf<TElem, Idx>(dev, extent);
    alpaka::test::iotaFillView(queue, buf);

    auto const mds = alpaka::experimental::getMdSpan(buf);
    alpaka::test::KernelExecutionFixture<Acc> fixture(Vec::ones());
    CHECK(fixture(
        [=] ALPAKA_FN_ACC(Acc const&, bool* success) NOEXCEPT_UNLESS_NVCC
        {
            auto counter = 0;
            alpaka::meta::ndLoopIncIdx(
                extent,
                [&](auto ind)
                {
                    [[maybe_unused]] auto const a = alpaka::toArray(ind);
                    ALPAKA_CHECK(*success, mds(a) == counter);
                    counter++;
                });
        }));

    auto extentTransposed = extent;
    std::reverse(extentTransposed.begin(), extentTransposed.end());
    auto const mdst = alpaka::experimental::getMdSpanTransposed(buf);
    CHECK(fixture(
        [=] ALPAKA_FN_ACC(Acc const&, bool* success) NOEXCEPT_UNLESS_NVCC
        {
            auto counter = 0;
            alpaka::meta::ndLoop(
                make_reverse_index_sequence<Dim::value>{},
                extentTransposed,
                [&](auto ind)
                {
                    [[maybe_unused]] auto const a = alpaka::toArray(ind);
                    ALPAKA_CHECK(*success, mdst(a) == counter);
                    counter++;
                });
        }));
}

TEMPLATE_LIST_TEST_CASE("submdspan", "[memView]", alpaka::test::TestAccs)
{
    using TElem = int;

    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const platform = alpaka::Platform<Dev>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Dev, alpaka::Blocking>(dev);

    auto const extent = alpaka::Vec{2, 3}; // 2 rows, 3 cols
    auto buf = alpaka::allocBuf<TElem, int>(dev, extent);
    alpaka::test::iotaFillView(queue, buf);

    auto const mds = alpaka::experimental::getMdSpan(buf);
    alpaka::test::KernelExecutionFixture<Acc> fixture(Vec::ones());
    CHECK(fixture(
        [=] ALPAKA_FN_ACC(Acc const&, bool* success) NOEXCEPT_UNLESS_NVCC
        {
            ALPAKA_CHECK(*success, mds(0, 0) == 0);
            ALPAKA_CHECK(*success, mds(0, 1) == 1);
            ALPAKA_CHECK(*success, mds(0, 2) == 2);
            ALPAKA_CHECK(*success, mds(1, 0) == 3);
            ALPAKA_CHECK(*success, mds(1, 1) == 4);
            ALPAKA_CHECK(*success, mds(1, 2) == 5);

            auto elem_1_2 = alpaka::experimental::submdspan(mds, 1, 2);
            ALPAKA_CHECK(*success, elem_1_2() == 5);

            auto row0 = alpaka::experimental::submdspan(mds, 0, alpaka::experimental::full_extent);
            ALPAKA_CHECK(*success, row0(0) == 0);
            ALPAKA_CHECK(*success, row0(1) == 1);
            ALPAKA_CHECK(*success, row0(2) == 2);

            auto row1 = alpaka::experimental::submdspan(mds, 1, alpaka::experimental::full_extent);
            ALPAKA_CHECK(*success, row1(0) == 3);
            ALPAKA_CHECK(*success, row1(1) == 4);
            ALPAKA_CHECK(*success, row1(2) == 5);

            auto col0 = alpaka::experimental::submdspan(mds, alpaka::experimental::full_extent, 0);
            ALPAKA_CHECK(*success, col0(0) == 0);
            ALPAKA_CHECK(*success, col0(1) == 3);

            auto col1 = alpaka::experimental::submdspan(mds, alpaka::experimental::full_extent, 1);
            ALPAKA_CHECK(*success, col1(0) == 1);
            ALPAKA_CHECK(*success, col1(1) == 4);

            auto col2 = alpaka::experimental::submdspan(mds, alpaka::experimental::full_extent, 2);
            ALPAKA_CHECK(*success, col2(0) == 2);
            ALPAKA_CHECK(*success, col2(1) == 5);

            auto matrix = alpaka::experimental::submdspan(
                mds,
                alpaka::experimental::full_extent,
                alpaka::experimental::full_extent);
            ALPAKA_CHECK(*success, matrix(0, 0) == 0);
            ALPAKA_CHECK(*success, matrix(0, 1) == 1);
            ALPAKA_CHECK(*success, matrix(0, 2) == 2);
            ALPAKA_CHECK(*success, matrix(1, 0) == 3);
            ALPAKA_CHECK(*success, matrix(1, 1) == 4);
            ALPAKA_CHECK(*success, matrix(1, 2) == 5);
        }));
}
#endif
