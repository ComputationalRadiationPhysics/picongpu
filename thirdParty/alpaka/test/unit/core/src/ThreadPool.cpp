/* Copyright 2023 Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/ThreadPool.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("threadpool", "[core]")
{
    alpaka::core::detail::ThreadPool tp{2};

#if defined(ALPAKA_COMP_ICPX) && ALPAKA_COMP_ICPX >= ALPAKA_VERSION_NUMBER(2025, 3, 0)                                \
    && ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2026, 0, 0)
    // This triggers a false positive with icpx 2025.3.x.
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#endif
    auto f1 = tp.enqueueTask([] { throw std::runtime_error("42"); });
    auto f2 = tp.enqueueTask([] { throw 42; });
    auto f3 = tp.enqueueTask([]() noexcept {});
#if defined(ALPAKA_COMP_ICPX) && ALPAKA_COMP_ICPX >= ALPAKA_VERSION_NUMBER(2025, 3, 0)                                \
    && ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2026, 0, 0)
#    pragma clang diagnostic pop
#endif

    CHECK_THROWS_AS(f1.get(), std::runtime_error);

#ifndef ALPAKA_USES_TSAN
    try
    {
        f2.get();
    }
    catch(int i)
    {
        CHECK(i == 42);
    }
#endif

    CHECK_NOTHROW(f3.get());
}
