/* Copyright 2023 Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/CallbackThread.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("callbackthread", "[core]")
{
    alpaka::core::CallbackThread cbt;

#if defined(ALPAKA_COMP_ICPX) && ALPAKA_COMP_ICPX >= ALPAKA_VERSION_NUMBER(2025, 3, 0)                                \
    && ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2026, 0, 0)
    // This triggers a false positive with icpx 2025.3.x.
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#endif
    auto f1 = cbt.submit([] { throw std::runtime_error("42"); });
    auto f2 = cbt.submit([] { throw 42; });
    auto f3 = cbt.submit([]() noexcept {});
#if defined(ALPAKA_COMP_ICPX) && ALPAKA_COMP_ICPX >= ALPAKA_VERSION_NUMBER(2025, 3, 0)                                \
    && ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2026, 0, 0)
#    pragma clang diagnostic pop
#endif

    CHECK_THROWS_AS(f1.get(), std::runtime_error);

    try
    {
        f2.get();
    }
    catch(int i)
    {
        CHECK(i == 42);
    }

    CHECK_NOTHROW(f3.get());
}

TEST_CASE("callbackthread_copyable_task", "[core]")
{
    alpaka::core::CallbackThread cbt;

    bool flag = false;
    auto task = [&] { flag = true; };
    cbt.submit(task).get(); // lvalue
    CHECK(flag == true);

    flag = false;
    cbt.submit(std::move(task)).get(); // xvalue
    CHECK(flag == true);

    flag = false;
    cbt.submit([&] { flag = true; }).get(); // prvalue
    CHECK(flag == true);
}

TEST_CASE("callbackthread_non_copyable_task", "[core]")
{
    alpaka::core::CallbackThread cbt;

    bool flag = false;
    auto task = [&, dummy = std::unique_ptr<int>{nullptr}] { flag = true; };
    cbt.submit(std::move(task)).get(); // xvalue
    CHECK(flag == true);

    flag = false;
    cbt.submit([&] { flag = true; }).get(); // prvalue
    CHECK(flag == true);
}
