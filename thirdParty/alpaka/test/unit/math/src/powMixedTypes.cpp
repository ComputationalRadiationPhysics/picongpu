/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber,
 *                Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Defines.hpp"

#include <alpaka/math/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

//! Convert the given real or complex input type to the given real or complex output type avoiding warnings.
//! This conversion is surprisingly tricky to do in a way that no compiler complains.
//! In principle it could be accomplished by a constexpr function, but in practice that turned out not possible.
//! The general implementation does direct initialization, works for pairs of types supporting it.
template<typename TInput, typename TOutput, typename TSfinae = void>
struct Convert
{
    ALPAKA_FN_ACC auto operator()(TInput const arg) const
    {
        return TOutput{arg};
    }
};

//! Specialization for real -> real conversion
template<typename TInput, typename TOutput>
struct Convert<TInput, TOutput, std::enable_if_t<std::is_floating_point_v<TOutput>>>
{
    ALPAKA_FN_ACC auto operator()(TInput const arg) const
    {
        return static_cast<TOutput>(arg);
    }
};

//! Specialization for real -> complex conversion
template<typename TInput, typename TOutputValueType>
struct Convert<TInput, alpaka::Complex<TOutputValueType>, std::enable_if_t<std::is_floating_point_v<TInput>>>
{
    ALPAKA_FN_ACC auto operator()(TInput const arg) const
    {
        return alpaka::Complex<TOutputValueType>{static_cast<TOutputValueType>(arg)};
    }
};

template<typename TExpected>
struct PowMixedTypesTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TArg1, typename TArg2>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, TArg1 const arg1, TArg2 const arg2) const -> void
    {
        auto expected = alpaka::math::pow(acc, Convert<TArg1, TExpected>{}(arg1), Convert<TArg2, TExpected>{}(arg2));
        auto actual = alpaka::math::pow(acc, arg1, arg2);
        ALPAKA_CHECK(*success, mathtest::almost_equal(acc, expected, actual, 1));
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("powMixedTypes", "[powMixedTypes]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PowMixedTypesTestKernel<float> kernelFloat;
    PowMixedTypesTestKernel<double> kernelDouble;
    PowMixedTypesTestKernel<alpaka::Complex<float>> kernelComplexFloat;
    PowMixedTypesTestKernel<alpaka::Complex<double>> kernelComplexDouble;

    float const floatArg = 0.35f;
    double const doubleArg = 0.24;
    alpaka::Complex<float> floatComplexArg{0.35f, -0.24f};
    alpaka::Complex<double> doubleComplexArg{0.35, -0.24};

    // all combinations of pow(real, real)
    REQUIRE(fixture(kernelFloat, floatArg, floatArg));
    REQUIRE(fixture(kernelDouble, floatArg, doubleArg));
    REQUIRE(fixture(kernelDouble, doubleArg, floatArg));
    REQUIRE(fixture(kernelDouble, doubleArg, doubleArg));

    // all combinations of pow(real, complex)
    REQUIRE(fixture(kernelComplexFloat, floatArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, floatArg, doubleComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleArg, doubleComplexArg));

    // all combinations of pow(complex, real)
    REQUIRE(fixture(kernelComplexFloat, floatComplexArg, floatArg));
    REQUIRE(fixture(kernelComplexDouble, floatComplexArg, doubleArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, floatArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, doubleArg));

    // all combinations of pow(complex, complex)
    REQUIRE(fixture(kernelComplexFloat, floatComplexArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, floatComplexArg, doubleComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, doubleComplexArg));
}
