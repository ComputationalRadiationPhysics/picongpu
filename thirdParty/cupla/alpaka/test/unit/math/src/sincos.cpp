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

struct SinCosTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename FP>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, FP const arg) const -> void
    {
        // if arg is hardcoded then compiler can optimize it out
        // (PTX kernel (float) was just empty)
        FP check_sin = alpaka::math::sin(acc, arg);
        FP check_cos = alpaka::math::cos(acc, arg);
        auto result_sin = FP{0};
        auto result_cos = FP{0};
        alpaka::math::sincos(acc, arg, result_sin, result_cos);
        ALPAKA_CHECK(
            *success,
            mathtest::almost_equal(acc, result_sin, check_sin, 1)
                && mathtest::almost_equal(acc, result_cos, check_cos, 1));
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("sincos", "[sincos]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    SinCosTestKernel kernel;

    REQUIRE(fixture(kernel, 0.42f)); // float
    REQUIRE(fixture(kernel, 0.42)); // double
    REQUIRE(fixture(kernel, alpaka::Complex<float>{0.35f, -0.24f})); // complex float
    REQUIRE(fixture(kernel, alpaka::Complex<double>{0.35, -0.24})); // complex double
}
