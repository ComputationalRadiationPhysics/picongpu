/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Buffer.hpp"
#include "DataGen.hpp"
#include "Defines.hpp"
#include "Functor.hpp"

#include <alpaka/math/Complex.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

namespace custom
{
    enum Custom
    {
        Abs,
        Acos,
        Acosh,
        Arg,
        Asin,
        Asinh,
        Atan,
        Atanh,
        Atan2,
        Cbrt,
        Ceil,
        Conj,
        Copysign,
        Cos,
        Cosh,
        Erf,
        Exp,
        Floor,
        Fma,
        Fmod,
        Log,
        Log2,
        Log10,
        Max,
        Min,
        Pow,
        Remainder,
        Round,
        Lround,
        Llround,
        Rsqrt,
        Sin,
        Sinh,
        Sincos,
        Sqrt,
        Tan,
        Tanh,
        Trunc,

        Arg1 = 1024,
        Arg2 = 2048,
        Arg3 = 4096,
    };

    ALPAKA_FN_HOST_ACC auto abs(Custom c);

    ALPAKA_FN_HOST_ACC auto abs(Custom c)
    {
        return Custom::Abs | c;
    }

    ALPAKA_FN_HOST_ACC auto acos(Custom c);

    ALPAKA_FN_HOST_ACC auto acos(Custom c)
    {
        return Custom::Acos | c;
    }

    ALPAKA_FN_HOST_ACC auto acosh(Custom c);

    ALPAKA_FN_HOST_ACC auto acosh(Custom c)
    {
        return Custom::Acosh | c;
    }

    ALPAKA_FN_HOST_ACC auto arg(Custom c);

    ALPAKA_FN_HOST_ACC auto arg(Custom c)
    {
        return Custom::Arg | c;
    }

    ALPAKA_FN_HOST_ACC auto asin(Custom c);

    ALPAKA_FN_HOST_ACC auto asin(Custom c)
    {
        return Custom::Asin | c;
    }

    ALPAKA_FN_HOST_ACC auto asinh(Custom c);

    ALPAKA_FN_HOST_ACC auto asinh(Custom c)
    {
        return Custom::Asinh | c;
    }

    ALPAKA_FN_HOST_ACC auto atan(Custom c);

    ALPAKA_FN_HOST_ACC auto atan(Custom c)
    {
        return Custom::Atan | c;
    }

    ALPAKA_FN_HOST_ACC auto atanh(Custom c);

    ALPAKA_FN_HOST_ACC auto atanh(Custom c)
    {
        return Custom::Atanh | c;
    }

    ALPAKA_FN_HOST_ACC auto atan2(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto atan2(Custom a, Custom b)
    {
        return Custom::Atan2 | a | b;
    }

    ALPAKA_FN_HOST_ACC auto cbrt(Custom c);

    ALPAKA_FN_HOST_ACC auto cbrt(Custom c)
    {
        return Custom::Cbrt | c;
    }

    ALPAKA_FN_HOST_ACC auto ceil(Custom c);

    ALPAKA_FN_HOST_ACC auto ceil(Custom c)
    {
        return Custom::Ceil | c;
    }

    ALPAKA_FN_HOST_ACC auto conj(Custom c);

    ALPAKA_FN_HOST_ACC auto conj(Custom c)
    {
        return Custom::Conj | c;
    }

    ALPAKA_FN_HOST_ACC auto copysign(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto copysign(Custom a, Custom b)
    {
        return Custom::Copysign | a | b;
    }

    ALPAKA_FN_HOST_ACC auto cos(Custom c);

    ALPAKA_FN_HOST_ACC auto cos(Custom c)
    {
        return Custom::Cos | c;
    }

    ALPAKA_FN_HOST_ACC auto cosh(Custom c);

    ALPAKA_FN_HOST_ACC auto cosh(Custom c)
    {
        return Custom::Cosh | c;
    }

    ALPAKA_FN_HOST_ACC auto erf(Custom c);

    ALPAKA_FN_HOST_ACC auto erf(Custom c)
    {
        return Custom::Erf | c;
    }

    ALPAKA_FN_HOST_ACC auto exp(Custom c);

    ALPAKA_FN_HOST_ACC auto exp(Custom c)
    {
        return Custom::Exp | c;
    }

    ALPAKA_FN_HOST_ACC auto floor(Custom c);

    ALPAKA_FN_HOST_ACC auto floor(Custom c)
    {
        return Custom::Floor | c;
    }

    ALPAKA_FN_HOST_ACC auto fma(Custom a, Custom b, Custom c);

    ALPAKA_FN_HOST_ACC auto fma(Custom a, Custom b, Custom c)
    {
        return Custom::Fma | a | b | c;
    }

    ALPAKA_FN_HOST_ACC auto fmod(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto fmod(Custom a, Custom b)
    {
        return Custom::Fmod | a | b;
    }

    ALPAKA_FN_HOST_ACC auto log(Custom c);

    ALPAKA_FN_HOST_ACC auto log(Custom c)
    {
        return Custom::Log | c;
    }

    ALPAKA_FN_HOST_ACC auto log2(Custom c);

    ALPAKA_FN_HOST_ACC auto log2(Custom c)
    {
        return Custom::Log2 | c;
    }

    ALPAKA_FN_HOST_ACC auto log10(Custom c);

    ALPAKA_FN_HOST_ACC auto log10(Custom c)
    {
        return Custom::Log10 | c;
    }

    ALPAKA_FN_HOST_ACC auto max(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto max(Custom a, Custom b)
    {
        return Custom::Max | a | b;
    }

    ALPAKA_FN_HOST_ACC auto min(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto min(Custom a, Custom b)
    {
        return Custom::Min | a | b;
    }

    ALPAKA_FN_HOST_ACC auto pow(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto pow(Custom a, Custom b)
    {
        return Custom::Pow | a | b;
    }

    ALPAKA_FN_HOST_ACC auto remainder(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto remainder(Custom a, Custom b)
    {
        return Custom::Remainder | a | b;
    }

    ALPAKA_FN_HOST_ACC auto round(Custom c);

    ALPAKA_FN_HOST_ACC auto round(Custom c)
    {
        return Custom::Round | c;
    }

    ALPAKA_FN_HOST_ACC auto lround(Custom c);

    ALPAKA_FN_HOST_ACC auto lround(Custom c)
    {
        return Custom::Lround | c;
    }

    ALPAKA_FN_HOST_ACC auto llround(Custom c);

    ALPAKA_FN_HOST_ACC auto llround(Custom c)
    {
        return Custom::Llround | c;
    }

    ALPAKA_FN_HOST_ACC auto rsqrt(Custom c);

    ALPAKA_FN_HOST_ACC auto rsqrt(Custom c)
    {
        return Custom::Rsqrt | c;
    }

    ALPAKA_FN_HOST_ACC auto sin(Custom c);

    ALPAKA_FN_HOST_ACC auto sin(Custom c)
    {
        return Custom::Sin | c;
    }

    ALPAKA_FN_HOST_ACC auto sinh(Custom c);

    ALPAKA_FN_HOST_ACC auto sinh(Custom c)
    {
        return Custom::Sinh | c;
    }

    ALPAKA_FN_HOST_ACC void sincos(Custom c, Custom& a, Custom& b);

    ALPAKA_FN_HOST_ACC void sincos(Custom c, Custom& a, Custom& b)
    {
        a = static_cast<Custom>(Custom::Sincos | c | Custom::Arg2);
        b = static_cast<Custom>(Custom::Sincos | c | Custom::Arg3);
    }

    ALPAKA_FN_HOST_ACC auto sqrt(Custom c);

    ALPAKA_FN_HOST_ACC auto sqrt(Custom c)
    {
        return Custom::Sqrt | c;
    }

    ALPAKA_FN_HOST_ACC auto tan(Custom c);

    ALPAKA_FN_HOST_ACC auto tan(Custom c)
    {
        return Custom::Tan | c;
    }

    ALPAKA_FN_HOST_ACC auto tanh(Custom c);

    ALPAKA_FN_HOST_ACC auto tanh(Custom c)
    {
        return Custom::Tanh | c;
    }

    ALPAKA_FN_HOST_ACC auto trunc(Custom c);

    ALPAKA_FN_HOST_ACC auto trunc(Custom c)
    {
        return Custom::Trunc | c;
    }
} // namespace custom

struct AdlKernel
{
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, bool* success) const noexcept
    {
        using custom::Custom;

        ALPAKA_CHECK(*success, alpaka::math::abs(acc, Custom::Arg1) == (Custom::Abs | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::acos(acc, Custom::Arg1) == (Custom::Acos | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::acosh(acc, Custom::Arg1) == (Custom::Acosh | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::arg(acc, Custom::Arg1) == (Custom::Arg | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::asin(acc, Custom::Arg1) == (Custom::Asin | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::asinh(acc, Custom::Arg1) == (Custom::Asinh | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::atan(acc, Custom::Arg1) == (Custom::Atan | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::atanh(acc, Custom::Arg1) == (Custom::Atanh | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::cbrt(acc, Custom::Arg1) == (Custom::Cbrt | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::ceil(acc, Custom::Arg1) == (Custom::Ceil | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::conj(acc, Custom::Arg1) == (Custom::Conj | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::cos(acc, Custom::Arg1) == (Custom::Cos | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::cosh(acc, Custom::Arg1) == (Custom::Cosh | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::erf(acc, Custom::Arg1) == (Custom::Erf | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::exp(acc, Custom::Arg1) == (Custom::Exp | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::floor(acc, Custom::Arg1) == (Custom::Floor | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::log(acc, Custom::Arg1) == (Custom::Log | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::log2(acc, Custom::Arg1) == (Custom::Log2 | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::log10(acc, Custom::Arg1) == (Custom::Log10 | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::round(acc, Custom::Arg1) == (Custom::Round | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::lround(acc, Custom::Arg1) == (Custom::Lround | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::llround(acc, Custom::Arg1) == (Custom::Llround | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::rsqrt(acc, Custom::Arg1) == (Custom::Rsqrt | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::sin(acc, Custom::Arg1) == (Custom::Sin | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::sinh(acc, Custom::Arg1) == (Custom::Sinh | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::sqrt(acc, Custom::Arg1) == (Custom::Sqrt | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::tan(acc, Custom::Arg1) == (Custom::Tan | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::tanh(acc, Custom::Arg1) == (Custom::Tanh | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::trunc(acc, Custom::Arg1) == (Custom::Trunc | Custom::Arg1));

        ALPAKA_CHECK(
            *success,
            alpaka::math::atan2(acc, Custom::Arg1, Custom::Arg2) == (Custom::Atan2 | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::copysign(acc, Custom::Arg1, Custom::Arg2)
                == (Custom::Copysign | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::fmod(acc, Custom::Arg1, Custom::Arg2) == (Custom::Fmod | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::max(acc, Custom::Arg1, Custom::Arg2) == (Custom::Max | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::min(acc, Custom::Arg1, Custom::Arg2) == (Custom::Min | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::pow(acc, Custom::Arg1, Custom::Arg2) == (Custom::Pow | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::remainder(acc, Custom::Arg1, Custom::Arg2)
                == (Custom::Remainder | Custom::Arg1 | Custom::Arg2));

        ALPAKA_CHECK(
            *success,
            alpaka::math::fma(acc, Custom::Arg1, Custom::Arg2, Custom::Arg3)
                == (Custom::Fma | Custom::Arg1 | Custom::Arg2 | Custom::Arg3));

        Custom a, b;
        alpaka::math::sincos(acc, Custom::Arg1, a, b);
        ALPAKA_CHECK(*success, a == (Custom::Sincos | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(*success, b == (Custom::Sincos | Custom::Arg1 | Custom::Arg3));
    }
};

TEMPLATE_LIST_TEST_CASE("mathOps", "[math] [operator] [adl]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    auto fixture = alpaka::test::KernelExecutionFixture<Acc>{alpaka::Vec<Dim, Idx>::ones()};
    REQUIRE(fixture(AdlKernel{}));
}
