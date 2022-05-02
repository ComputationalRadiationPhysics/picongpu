/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber, Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/math/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <type_traits>

// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<typename TAcc, typename FP>
ALPAKA_FN_ACC auto almost_equal(TAcc const& acc, FP x, FP y, int ulp)
    -> std::enable_if_t<!std::numeric_limits<FP>::is_integer, bool>
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return alpaka::math::abs(acc, x - y)
        <= std::numeric_limits<FP>::epsilon() * alpaka::math::abs(acc, x + y) * static_cast<FP>(ulp)
        // unless the result is subnormal
        || alpaka::math::abs(acc, x - y) < std::numeric_limits<FP>::min();
}

//! Version for alpaka::Complex
template<typename TAcc, typename FP>
ALPAKA_FN_ACC bool almost_equal(TAcc const& acc, alpaka::Complex<FP> x, alpaka::Complex<FP> y, int ulp)
{
    return almost_equal(acc, x.real(), y.real(), ulp) && almost_equal(acc, x.imag(), y.imag(), ulp);
}

class SinCosTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename FP>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, FP const arg) const -> void
    {
        // if arg is hardcoded then compiler can optimize it out
        // (PTX kernel (float) was just empty)
        FP check_sin = alpaka::math::sin(acc, arg);
        FP check_cos = alpaka::math::cos(acc, arg);
        FP result_sin = 0.;
        FP result_cos = 0.;
        alpaka::math::sincos(acc, arg, result_sin, result_cos);
        ALPAKA_CHECK(
            *success,
            almost_equal(acc, result_sin, check_sin, 1) && almost_equal(acc, result_cos, check_cos, 1));
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
