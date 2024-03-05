/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Functor.hpp"
#include "TestTemplate.hpp"

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <tuple>

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

// This file only has unit tests for real numbers in order to split the tests between object files
using FunctorsReal = alpaka::meta::
    Concatenate<mathtest::UnaryFunctorsReal, mathtest::BinaryFunctorsReal, mathtest::TernaryFunctorsReal>;
using TestAccFunctorTuplesReal = alpaka::meta::CartesianProduct<std::tuple, TestAccs, FunctorsReal>;

TEMPLATE_LIST_TEST_CASE("mathOpsDouble", "[math] [operator]", TestAccFunctorTuplesReal)
{
    // Same as "mathOpsFloat" template test, but for double. See detailed explanation there.
    using Acc = std::tuple_element_t<0u, TestType>;
    using Functor = std::tuple_element_t<1u, TestType>;
    auto testTemplate = mathtest::TestTemplate<Acc, Functor>{};
    testTemplate.template operator()<double>();
}
