/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Functor.hpp"
#include "TestTemplate.hpp"

#include <alpaka/math/Complex.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <tuple>
#include <type_traits>

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

// This file only has unit tests for complex numbers in order to split the tests between object files and save compiler
// memory. For the same reason single- and double-precision are done separately and not wrapped into a common template.
using FunctorsComplex = alpaka::meta::Concatenate<mathtest::UnaryFunctorsComplex, mathtest::BinaryFunctorsComplex>;
using TestAccFunctorTuplesComplex = alpaka::meta::CartesianProduct<std::tuple, TestAccs, FunctorsComplex>;

TEMPLATE_LIST_TEST_CASE("mathOpsComplexDouble", "[math] [operator]", TestAccFunctorTuplesComplex)
{
    // Same as "mathOpsFloat" template test, but for complex double. See detailed explanation there.
    using Acc = std::tuple_element_t<0u, TestType>;
    using Functor = std::tuple_element_t<1u, TestType>;
    auto testTemplate = mathtest::TestTemplate<Acc, Functor>{};
    testTemplate.template operator()<alpaka::Complex<double>>();
}

TEST_CASE("mathArrayOrientedComplexDouble", "[array-oriented]")
{
    // Ensure that our implementation matches the behaviour of std::complex with regard to array-oriented access.
    // See https://en.cppreference.com/w/cpp/numeric/complex - Array-oriented access - for more information.
    auto const complex_alpaka = alpaka::Complex<double>{42., 42.};
    auto const complex_std = std::complex<double>{42., 42.};

    auto const real_alpaka = reinterpret_cast<double const(&)[2]>(complex_alpaka)[0];
    auto const real_std = reinterpret_cast<double const(&)[2]>(complex_std)[0];
    REQUIRE(alpaka::math::floatEqualExactNoWarning(real_alpaka, real_std));

    auto const imag_alpaka = reinterpret_cast<double const(&)[2]>(complex_alpaka)[1];
    auto const imag_std = reinterpret_cast<double const(&)[2]>(complex_std)[1];
    REQUIRE(alpaka::math::floatEqualExactNoWarning(imag_alpaka, imag_std));
}

TEST_CASE("mathPaddingComplexDouble", "[padding]")
{
    // Ensure that we don't accidentally introduce padding
    STATIC_REQUIRE(sizeof(alpaka::Complex<double>) == 2 * sizeof(double));
}
