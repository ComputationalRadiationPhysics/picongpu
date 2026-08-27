/* Copyright 2025 Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/dev/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("tagToDevice", "[dev]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Tag = alpaka::AccToTag<Acc>;

    STATIC_REQUIRE(std::is_same_v<alpaka::Dev<Tag>, alpaka::Dev<Acc>>);
}
