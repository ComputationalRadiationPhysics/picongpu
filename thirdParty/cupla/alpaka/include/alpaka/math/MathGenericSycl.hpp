/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/math/Complex.hpp>
#    include <alpaka/math/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

//! The mathematical operation specifics.
namespace alpaka::experimental::math
{
    //! The SYCL abs.
    class AbsGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAbs, AbsGenericSycl>
    {
    };

    //! The SYCL acos.
    class AcosGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAcos, AcosGenericSycl>
    {
    };

    //! The SYCL acosh.
    class AcoshGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAcosh, AcoshGenericSycl>
    {
    };

    //! The SYCL arg.
    class ArgGenericSycl : public concepts::Implements<alpaka::math::ConceptMathArg, ArgGenericSycl>
    {
    };

    //! The SYCL asin.
    class AsinGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAsin, AsinGenericSycl>
    {
    };

    //! The SYCL asinh.
    class AsinhGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAsinh, AsinhGenericSycl>
    {
    };

    //! The SYCL atan.
    class AtanGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAtan, AtanGenericSycl>
    {
    };

    //! The SYCL atanh.
    class AtanhGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAtanh, AtanhGenericSycl>
    {
    };

    //! The SYCL atan2.
    class Atan2GenericSycl : public concepts::Implements<alpaka::math::ConceptMathAtan2, Atan2GenericSycl>
    {
    };

    //! The SYCL cbrt.
    class CbrtGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCbrt, CbrtGenericSycl>
    {
    };

    //! The SYCL ceil.
    class CeilGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCeil, CeilGenericSycl>
    {
    };

    //! The SYCL conj.
    class ConjGenericSycl : public concepts::Implements<alpaka::math::ConceptMathConj, ConjGenericSycl>
    {
    };

    //! The SYCL cos.
    class CosGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCos, CosGenericSycl>
    {
    };

    //! The SYCL cosh.
    class CoshGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCosh, CoshGenericSycl>
    {
    };

    //! The SYCL erf.
    class ErfGenericSycl : public concepts::Implements<alpaka::math::ConceptMathErf, ErfGenericSycl>
    {
    };

    //! The SYCL exp.
    class ExpGenericSycl : public concepts::Implements<alpaka::math::ConceptMathExp, ExpGenericSycl>
    {
    };

    //! The SYCL floor.
    class FloorGenericSycl : public concepts::Implements<alpaka::math::ConceptMathFloor, FloorGenericSycl>
    {
    };

    //! The SYCL fmod.
    class FmodGenericSycl : public concepts::Implements<alpaka::math::ConceptMathFmod, FmodGenericSycl>
    {
    };

    //! The SYCL isfinite.
    class IsfiniteGenericSycl : public concepts::Implements<alpaka::math::ConceptMathIsfinite, IsfiniteGenericSycl>
    {
    };

    //! The SYCL isfinite.
    class IsinfGenericSycl : public concepts::Implements<alpaka::math::ConceptMathIsinf, IsinfGenericSycl>
    {
    };

    //! The SYCL isnan.
    class IsnanGenericSycl : public concepts::Implements<alpaka::math::ConceptMathIsnan, IsnanGenericSycl>
    {
    };

    //! The SYCL log.
    class LogGenericSycl : public concepts::Implements<alpaka::math::ConceptMathLog, LogGenericSycl>
    {
    };

    //! The SYCL max.
    class MaxGenericSycl : public concepts::Implements<alpaka::math::ConceptMathMax, MaxGenericSycl>
    {
    };

    //! The SYCL min.
    class MinGenericSycl : public concepts::Implements<alpaka::math::ConceptMathMin, MinGenericSycl>
    {
    };

    //! The SYCL pow.
    class PowGenericSycl : public concepts::Implements<alpaka::math::ConceptMathPow, PowGenericSycl>
    {
    };

    //! The SYCL remainder.
    class RemainderGenericSycl : public concepts::Implements<alpaka::math::ConceptMathRemainder, RemainderGenericSycl>
    {
    };

    //! The SYCL round.
    class RoundGenericSycl : public concepts::Implements<alpaka::math::ConceptMathRound, RoundGenericSycl>
    {
    };

    //! The SYCL rsqrt.
    class RsqrtGenericSycl : public concepts::Implements<alpaka::math::ConceptMathRsqrt, RsqrtGenericSycl>
    {
    };

    //! The SYCL sin.
    class SinGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSin, SinGenericSycl>
    {
    };

    //! The SYCL sinh.
    class SinhGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSinh, SinhGenericSycl>
    {
    };

    //! The SYCL sincos.
    class SinCosGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSinCos, SinCosGenericSycl>
    {
    };

    //! The SYCL sqrt.
    class SqrtGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSqrt, SqrtGenericSycl>
    {
    };

    //! The SYCL tan.
    class TanGenericSycl : public concepts::Implements<alpaka::math::ConceptMathTan, TanGenericSycl>
    {
    };

    //! The SYCL tanh.
    class TanhGenericSycl : public concepts::Implements<alpaka::math::ConceptMathTanh, TanhGenericSycl>
    {
    };

    //! The SYCL trunc.
    class TruncGenericSycl : public concepts::Implements<alpaka::math::ConceptMathTrunc, TruncGenericSycl>
    {
    };

    //! The SYCL math trait specializations.
    class MathGenericSycl
        : public AbsGenericSycl
        , public AcosGenericSycl
        , public AcoshGenericSycl
        , public ArgGenericSycl
        , public AsinGenericSycl
        , public AsinhGenericSycl
        , public AtanGenericSycl
        , public AtanhGenericSycl
        , public Atan2GenericSycl
        , public CbrtGenericSycl
        , public CeilGenericSycl
        , public ConjGenericSycl
        , public CosGenericSycl
        , public CoshGenericSycl
        , public ErfGenericSycl
        , public ExpGenericSycl
        , public FloorGenericSycl
        , public FmodGenericSycl
        , public IsfiniteGenericSycl
        , public IsinfGenericSycl
        , public IsnanGenericSycl
        , public LogGenericSycl
        , public MaxGenericSycl
        , public MinGenericSycl
        , public PowGenericSycl
        , public RemainderGenericSycl
        , public RoundGenericSycl
        , public RsqrtGenericSycl
        , public SinGenericSycl
        , public SinhGenericSycl
        , public SinCosGenericSycl
        , public SqrtGenericSycl
        , public TanGenericSycl
        , public TanhGenericSycl
        , public TruncGenericSycl
    {
    };
} // namespace alpaka::experimental::math

namespace alpaka::math::trait
{
    //! The SYCL abs trait specialization.
    template<typename TArg>
    struct Abs<experimental::math::AbsGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
    {
        auto operator()(experimental::math::AbsGenericSycl const&, TArg const& arg)
        {
            if constexpr(std::is_integral_v<TArg>)
                return sycl::abs(arg);
            else if constexpr(std::is_floating_point_v<TArg>)
                return sycl::fabs(arg);
            else
                static_assert(!sizeof(TArg), "Unsupported data type");
        }
    };

    //! The SYCL acos trait specialization.
    template<typename TArg>
    struct Acos<experimental::math::AcosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::AcosGenericSycl const&, TArg const& arg)
        {
            return sycl::acos(arg);
        }
    };

    //! The SYCL acosh trait specialization.
    template<typename TArg>
    struct Acosh<experimental::math::AcoshGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::AcoshGenericSycl const&, TArg const& arg)
        {
            return sycl::acosh(arg);
        }
    };

    //! The SYCL arg trait specialization.
    template<typename TArgument>
    struct Arg<experimental::math::ArgGenericSycl, TArgument, std::enable_if_t<std::is_arithmetic_v<TArgument>>>
    {
        auto operator()(experimental::math::ArgGenericSycl const&, TArgument const& argument)
        {
            if constexpr(std::is_integral_v<TArgument>)
                return sycl::atan2(0.0, static_cast<double>(argument));
            else if constexpr(std::is_floating_point_v<TArgument>)
                return sycl::atan2(TArgument{0.0}, argument);
            else
                static_assert(!sizeof(TArgument), "Unsupported data type");
        }
    };

    //! The SYCL asin trait specialization.
    template<typename TArg>
    struct Asin<experimental::math::AsinGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::AsinGenericSycl const&, TArg const& arg)
        {
            return sycl::asin(arg);
        }
    };

    //! The SYCL asinh trait specialization.
    template<typename TArg>
    struct Asinh<experimental::math::AsinhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::AsinhGenericSycl const&, TArg const& arg)
        {
            return sycl::asinh(arg);
        }
    };

    //! The SYCL atan trait specialization.
    template<typename TArg>
    struct Atan<experimental::math::AtanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::AtanGenericSycl const&, TArg const& arg)
        {
            return sycl::atan(arg);
        }
    };

    //! The SYCL atanh trait specialization.
    template<typename TArg>
    struct Atanh<experimental::math::AtanhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::AtanhGenericSycl const&, TArg const& arg)
        {
            return sycl::atanh(arg);
        }
    };

    //! The SYCL atan2 trait specialization.
    template<typename Ty, typename Tx>
    struct Atan2<
        experimental::math::Atan2GenericSycl,
        Ty,
        Tx,
        std::enable_if_t<std::is_floating_point_v<Ty> && std::is_floating_point_v<Tx>>>
    {
        auto operator()(experimental::math::Atan2GenericSycl const&, Ty const& y, Tx const& x)
        {
            return sycl::atan2(y, x);
        }
    };

    //! The SYCL cbrt trait specialization.
    template<typename TArg>
    struct Cbrt<experimental::math::CbrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
    {
        auto operator()(experimental::math::CbrtGenericSycl const&, TArg const& arg)
        {
            if constexpr(std::is_integral_v<TArg>)
                return sycl::cbrt(static_cast<double>(arg)); // Mirror CUDA back-end and use double for ints
            else if constexpr(std::is_floating_point_v<TArg>)
                return sycl::cbrt(arg);
            else
                static_assert(!sizeof(TArg), "Unsupported data type");
        }
    };

    //! The SYCL ceil trait specialization.
    template<typename TArg>
    struct Ceil<experimental::math::CeilGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::CeilGenericSycl const&, TArg const& arg)
        {
            return sycl::ceil(arg);
        }
    };

    //! The SYCL conj trait specialization.
    template<typename TArg>
    struct Conj<experimental::math::ConjGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::ConjGenericSycl const&, TArg const& arg)
        {
            return Complex<TArg>{arg, TArg{0.0}};
        }
    };

    //! The SYCL cos trait specialization.
    template<typename TArg>
    struct Cos<experimental::math::CosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::CosGenericSycl const&, TArg const& arg)
        {
            return sycl::cos(arg);
        }
    };

    //! The SYCL cos trait specialization.
    template<typename TArg>
    struct Cosh<experimental::math::CoshGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::CoshGenericSycl const&, TArg const& arg)
        {
            return sycl::cosh(arg);
        }
    };

    //! The SYCL erf trait specialization.
    template<typename TArg>
    struct Erf<experimental::math::ErfGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::ErfGenericSycl const&, TArg const& arg)
        {
            return sycl::erf(arg);
        }
    };

    //! The SYCL exp trait specialization.
    template<typename TArg>
    struct Exp<experimental::math::ExpGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::ExpGenericSycl const&, TArg const& arg)
        {
            return sycl::exp(arg);
        }
    };

    //! The SYCL floor trait specialization.
    template<typename TArg>
    struct Floor<experimental::math::FloorGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::FloorGenericSycl const&, TArg const& arg)
        {
            return sycl::floor(arg);
        }
    };

    //! The SYCL fmod trait specialization.
    template<typename Tx, typename Ty>
    struct Fmod<
        experimental::math::FmodGenericSycl,
        Tx,
        Ty,
        std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
    {
        auto operator()(experimental::math::FmodGenericSycl const&, Tx const& x, Ty const& y)
        {
            return sycl::fmod(x, y);
        }
    };

    //! The SYCL isfinite trait specialization.
    template<typename TArg>
    struct Isfinite<experimental::math::IsfiniteGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::IsfiniteGenericSycl const&, TArg const& arg)
        {
            return sycl::isfinite(arg);
        }
    };

    //! The SYCL isinf trait specialization.
    template<typename TArg>
    struct Isinf<experimental::math::IsinfGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::IsinfGenericSycl const&, TArg const& arg)
        {
            return sycl::isinf(arg);
        }
    };

    //! The SYCL isnan trait specialization.
    template<typename TArg>
    struct Isnan<experimental::math::IsnanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::IsnanGenericSycl const&, TArg const& arg)
        {
            return sycl::isnan(arg);
        }
    };

    //! The SYCL log trait specialization.
    template<typename TArg>
    struct Log<experimental::math::LogGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::LogGenericSycl const&, TArg const& arg)
        {
            return sycl::log(arg);
        }
    };

    //! The SYCL max trait specialization.
    template<typename Tx, typename Ty>
    struct Max<
        experimental::math::MaxGenericSycl,
        Tx,
        Ty,
        std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
    {
        auto operator()(experimental::math::MaxGenericSycl const&, Tx const& x, Ty const& y)
        {
            if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                return sycl::max(x, y);
            else if constexpr(std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>)
                return sycl::fmax(x, y);
            else if constexpr(
                (std::is_floating_point_v<Tx> && std::is_integral_v<Ty>)
                || (std::is_integral_v<Tx> && std::is_floating_point_v<Ty>) )
                return sycl::fmax(static_cast<double>(x), static_cast<double>(y)); // mirror CUDA back-end
            else
                static_assert(!sizeof(Tx), "Unsupported data type");
        }
    };

    //! The SYCL min trait specialization.
    template<typename Tx, typename Ty>
    struct Min<
        experimental::math::MinGenericSycl,
        Tx,
        Ty,
        std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
    {
        auto operator()(experimental::math::MinGenericSycl const&, Tx const& x, Ty const& y)
        {
            if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                return sycl::min(x, y);
            else if constexpr(std::is_floating_point_v<Tx> || std::is_floating_point_v<Ty>)
                return sycl::fmin(x, y);
            else if constexpr(
                (std::is_floating_point_v<Tx> && std::is_integral_v<Ty>)
                || (std::is_integral_v<Tx> && std::is_floating_point_v<Ty>) )
                return sycl::fmin(static_cast<double>(x), static_cast<double>(y)); // mirror CUDA back-end
            else
                static_assert(!sizeof(Tx), "Unsupported data type");
        }
    };

    //! The SYCL pow trait specialization.
    template<typename TBase, typename TExp>
    struct Pow<
        experimental::math::PowGenericSycl,
        TBase,
        TExp,
        std::enable_if_t<std::is_floating_point_v<TBase> && std::is_floating_point_v<TExp>>>
    {
        auto operator()(experimental::math::PowGenericSycl const&, TBase const& base, TExp const& exp)
        {
            return sycl::pow(base, exp);
        }
    };

    //! The SYCL remainder trait specialization.
    template<typename Tx, typename Ty>
    struct Remainder<
        experimental::math::RemainderGenericSycl,
        Tx,
        Ty,
        std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
    {
        auto operator()(experimental::math::RemainderGenericSycl const&, Tx const& x, Ty const& y)
        {
            return sycl::remainder(x, y);
        }
    };

    //! The SYCL round trait specialization.
    template<typename TArg>
    struct Round<experimental::math::RoundGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::RoundGenericSycl const&, TArg const& arg)
        {
            return sycl::round(arg);
        }
    };

    //! The SYCL lround trait specialization.
    template<typename TArg>
    struct Lround<experimental::math::RoundGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::RoundGenericSycl const&, TArg const& arg)
        {
            return static_cast<long>(sycl::round(arg));
        }
    };

    //! The SYCL llround trait specialization.
    template<typename TArg>
    struct Llround<experimental::math::RoundGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::RoundGenericSycl const&, TArg const& arg)
        {
            return static_cast<long long>(sycl::round(arg));
        }
    };

    //! The SYCL rsqrt trait specialization.
    template<typename TArg>
    struct Rsqrt<experimental::math::RsqrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
    {
        auto operator()(experimental::math::RsqrtGenericSycl const&, TArg const& arg)
        {
            if(std::is_floating_point_v<TArg>)
                return sycl::rsqrt(arg);
            else if(std::is_integral_v<TArg>)
                return sycl::rsqrt(static_cast<double>(arg)); // mirror CUDA back-end and use double for ints
            else
                static_assert(!sizeof(TArg), "Unsupported data type");
        }
    };

    //! The SYCL sin trait specialization.
    template<typename TArg>
    struct Sin<experimental::math::SinGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::SinGenericSycl const&, TArg const& arg)
        {
            return sycl::sin(arg);
        }
    };

    //! The SYCL sinh trait specialization.
    template<typename TArg>
    struct Sinh<experimental::math::SinhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::SinhGenericSycl const&, TArg const& arg)
        {
            return sycl::sinh(arg);
        }
    };

    //! The SYCL sincos trait specialization.
    template<typename TArg>
    struct SinCos<experimental::math::SinCosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(
            experimental::math::SinCosGenericSycl const&,
            TArg const& arg,
            TArg& result_sin,
            TArg& result_cos) -> void
        {
            result_sin = sycl::sincos(arg, &result_cos);
        }
    };

    //! The SYCL sqrt trait specialization.
    template<typename TArg>
    struct Sqrt<experimental::math::SqrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
    {
        auto operator()(experimental::math::SqrtGenericSycl const&, TArg const& arg)
        {
            if constexpr(std::is_floating_point_v<TArg>)
                return sycl::sqrt(arg);
            else if constexpr(std::is_integral_v<TArg>)
                return sycl::sqrt(static_cast<double>(arg)); // mirror CUDA back-end and use double for ints
        }
    };

    //! The SYCL tan trait specialization.
    template<typename TArg>
    struct Tan<experimental::math::TanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::TanGenericSycl const&, TArg const& arg)
        {
            return sycl::tan(arg);
        }
    };

    //! The SYCL tanh trait specialization.
    template<typename TArg>
    struct Tanh<experimental::math::TanhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::TanhGenericSycl const&, TArg const& arg)
        {
            return sycl::tanh(arg);
        }
    };

    //! The SYCL trunc trait specialization.
    template<typename TArg>
    struct Trunc<experimental::math::TruncGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(experimental::math::TruncGenericSycl const&, TArg const& arg)
        {
            return sycl::trunc(arg);
        }
    };
} // namespace alpaka::math::trait

#endif
