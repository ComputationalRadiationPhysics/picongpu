/* Copyright 2023 Antonio Di Pilato, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/traits/Traits.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("NativeHandle", "[handle]", alpaka::test::TestAccs)
{
    using Dev = alpaka::Dev<TestType>;

    auto const platformAcc = alpaka::Platform<TestType>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
    auto handle = alpaka::getNativeHandle(devAcc);

    STATIC_REQUIRE(std::is_same_v<alpaka::NativeHandle<Dev>, decltype(handle)>);
    // The SYCL backend does not use an int as the native handle type
#ifndef ALPAKA_ACC_SYCL_ENABLED
    STATIC_REQUIRE(std::is_same_v<alpaka::NativeHandle<Dev>, int>);
#endif
}
