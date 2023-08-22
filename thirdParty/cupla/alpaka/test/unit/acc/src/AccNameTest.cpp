/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <iostream>

TEMPLATE_LIST_TEST_CASE("getAccName", "[acc]", alpaka::test::TestAccs)
{
    std::cout << alpaka::getAccName<TestType>() << std::endl;
}
