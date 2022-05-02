/** Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "Functor.hpp"
#include "TestTemplate.hpp"

#include <alpaka/math/Complex.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

#include <complex>
#include <tuple>
#include <type_traits>

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

// This file only has unit tests for complex numbers in order to split the tests between object files and save compiler
// memory. For the same reason single- and double-precision are done separately and not wrapped into a common template.
using FunctorsComplex = alpaka::meta::
    Concatenate<alpaka::test::unit::math::UnaryFunctorsComplex, alpaka::test::unit::math::BinaryFunctorsComplex>;
using TestAccFunctorTuplesComplex = alpaka::meta::CartesianProduct<std::tuple, TestAccs, FunctorsComplex>;

TEMPLATE_LIST_TEST_CASE("mathOpsComplexFloat", "[math] [operator]", TestAccFunctorTuplesComplex)
{
    // Same as "mathOpsFloat" template test, but for complex float. See detailed explanation there.
    using Acc = std::tuple_element_t<0u, TestType>;
    using Functor = std::tuple_element_t<1u, TestType>;
    auto testTemplate = TestTemplate<Acc, Functor>{};
    testTemplate.template operator()<alpaka::Complex<float>>();
}

#ifdef __cpp_lib_is_layout_compatible
TEMPLATE_LIST_TEST_CASE("mathOpsComplexFloat", "[layout]", TestAccs)
{
    STATIC_REQUIRE(std::is_layout_compatible_v<alpaka::Complex<float>, std::complex<float>>);
}
#endif
