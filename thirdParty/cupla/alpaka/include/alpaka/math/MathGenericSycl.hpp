/* Copyright 2023 Jan Stephan, Sergei Bastrakov, René Widera, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/math/Complex.hpp"
#include "alpaka/math/Traits.hpp"

#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

//! The mathematical operation specifics.
namespace alpaka::math
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

    //! The SYCL copysign.
    class CopysignGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCopysign, CopysignGenericSycl>
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

    //! The SYCL fma.
    class FmaGenericSycl : public concepts::Implements<alpaka::math::ConceptMathFma, FmaGenericSycl>
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

    //! The SYCL log2.
    class Log2GenericSycl : public concepts::Implements<alpaka::math::ConceptMathLog2, Log2GenericSycl>
    {
    };

    //! The SYCL log10.
    class Log10GenericSycl : public concepts::Implements<alpaka::math::ConceptMathLog10, Log10GenericSycl>
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
        , public CopysignGenericSycl
        , public CosGenericSycl
        , public CoshGenericSycl
        , public ErfGenericSycl
        , public ExpGenericSycl
        , public FloorGenericSycl
        , public FmaGenericSycl
        , public FmodGenericSycl
        , public IsfiniteGenericSycl
        , public IsinfGenericSycl
        , public IsnanGenericSycl
        , public LogGenericSycl
        , public Log2GenericSycl
        , public Log10GenericSycl
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
} // namespace alpaka::math

namespace alpaka::math::trait
{
    //! The SYCL abs trait specialization.
    template<typename TArg>
    struct Abs<math::AbsGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
    {
        auto operator()(math::AbsGenericSycl const&, TArg const& arg)
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
    struct Acos<math::AcosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::AcosGenericSycl const&, TArg const& arg)
        {
            return sycl::acos(arg);
        }
    };

    //! The SYCL acosh trait specialization.
    template<typename TArg>
    struct Acosh<math::AcoshGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::AcoshGenericSycl const&, TArg const& arg)
        {
            return sycl::acosh(arg);
        }
    };

    //! The SYCL arg trait specialization.
    template<typename TArgument>
    struct Arg<math::ArgGenericSycl, TArgument, std::enable_if_t<std::is_arithmetic_v<TArgument>>>
    {
        auto operator()(math::ArgGenericSycl const&, TArgument const& argument)
        {
            if constexpr(std::is_integral_v<TArgument>)
                return sycl::atan2(0.0, static_cast<double>(argument));
            else if constexpr(std::is_floating_point_v<TArgument>)
                return sycl::atan2(static_cast<TArgument>(0.0), argument);
            else
                static_assert(!sizeof(TArgument), "Unsupported data type");
        }
    };

    //! The SYCL asin trait specialization.
    template<typename TArg>
    struct Asin<math::AsinGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::AsinGenericSycl const&, TArg const& arg)
        {
            return sycl::asin(arg);
        }
    };

    //! The SYCL asinh trait specialization.
    template<typename TArg>
    struct Asinh<math::AsinhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::AsinhGenericSycl const&, TArg const& arg)
        {
            return sycl::asinh(arg);
        }
    };

    //! The SYCL atan trait specialization.
    template<typename TArg>
    struct Atan<math::AtanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::AtanGenericSycl const&, TArg const& arg)
        {
            return sycl::atan(arg);
        }
    };

    //! The SYCL atanh trait specialization.
    template<typename TArg>
    struct Atanh<math::AtanhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::AtanhGenericSycl const&, TArg const& arg)
        {
            return sycl::atanh(arg);
        }
    };

    //! The SYCL atan2 trait specialization.
    template<typename Ty, typename Tx>
    struct Atan2<
        math::Atan2GenericSycl,
        Ty,
        Tx,
        std::enable_if_t<std::is_floating_point_v<Ty> && std::is_floating_point_v<Tx>>>
    {
        using TCommon = std::common_type_t<Ty, Tx>;

        auto operator()(math::Atan2GenericSycl const&, Ty const& y, Tx const& x)
        {
            return sycl::atan2(static_cast<TCommon>(y), static_cast<TCommon>(x));
        }
    };

    //! The SYCL cbrt trait specialization.
    template<typename TArg>
    struct Cbrt<math::CbrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
    {
        auto operator()(math::CbrtGenericSycl const&, TArg const& arg)
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
    struct Ceil<math::CeilGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::CeilGenericSycl const&, TArg const& arg)
        {
            return sycl::ceil(arg);
        }
    };

    //! The SYCL conj trait specialization.
    template<typename TArg>
    struct Conj<math::ConjGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::ConjGenericSycl const&, TArg const& arg)
        {
            return Complex<TArg>{arg, TArg{0.0}};
        }
    };

    //! The SYCL copysign trait specialization.
    template<typename TMag, typename TSgn>
    struct Copysign<
        math::CopysignGenericSycl,
        TMag,
        TSgn,
        std::enable_if_t<std::is_floating_point_v<TMag> && std::is_floating_point_v<TSgn>>>
    {
        using TCommon = std::common_type_t<TMag, TSgn>;

        auto operator()(math::CopysignGenericSycl const&, TMag const& y, TSgn const& x)
        {
            return sycl::copysign(static_cast<TCommon>(y), static_cast<TCommon>(x));
        }
    };

    //! The SYCL cos trait specialization.
    template<typename TArg>
    struct Cos<math::CosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::CosGenericSycl const&, TArg const& arg)
        {
            return sycl::cos(arg);
        }
    };

    //! The SYCL cos trait specialization.
    template<typename TArg>
    struct Cosh<math::CoshGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::CoshGenericSycl const&, TArg const& arg)
        {
            return sycl::cosh(arg);
        }
    };

    //! The SYCL erf trait specialization.
    template<typename TArg>
    struct Erf<math::ErfGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::ErfGenericSycl const&, TArg const& arg)
        {
            return sycl::erf(arg);
        }
    };

    //! The SYCL exp trait specialization.
    template<typename TArg>
    struct Exp<math::ExpGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::ExpGenericSycl const&, TArg const& arg)
        {
            return sycl::exp(arg);
        }
    };

    //! The SYCL floor trait specialization.
    template<typename TArg>
    struct Floor<math::FloorGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::FloorGenericSycl const&, TArg const& arg)
        {
            return sycl::floor(arg);
        }
    };

    //! The SYCL fma trait specialization.
    template<typename Tx, typename Ty, typename Tz>
    struct Fma<
        math::FmaGenericSycl,
        Tx,
        Ty,
        Tz,
        std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty> && std::is_floating_point_v<Tz>>>
    {
        auto operator()(math::FmaGenericSycl const&, Tx const& x, Ty const& y, Tz const& z)
        {
            return sycl::fma(x, y, z);
        }
    };

    //! The SYCL fmod trait specialization.
    template<typename Tx, typename Ty>
    struct Fmod<
        math::FmodGenericSycl,
        Tx,
        Ty,
        std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
    {
        using TCommon = std::common_type_t<Tx, Ty>;

        auto operator()(math::FmodGenericSycl const&, Tx const& x, Ty const& y)
        {
            return sycl::fmod(static_cast<TCommon>(x), static_cast<TCommon>(y));
        }
    };

    //! The SYCL isfinite trait specialization.
    template<typename TArg>
    struct Isfinite<math::IsfiniteGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::IsfiniteGenericSycl const&, TArg const& arg)
        {
            return static_cast<bool>(sycl::isfinite(arg));
        }
    };

    //! The SYCL isinf trait specialization.
    template<typename TArg>
    struct Isinf<math::IsinfGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::IsinfGenericSycl const&, TArg const& arg)
        {
            return static_cast<bool>(sycl::isinf(arg));
        }
    };

    //! The SYCL isnan trait specialization.
    template<typename TArg>
    struct Isnan<math::IsnanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::IsnanGenericSycl const&, TArg const& arg)
        {
            return static_cast<bool>(sycl::isnan(arg));
        }
    };

    //! The SYCL log trait specialization.
    template<typename TArg>
    struct Log<math::LogGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::LogGenericSycl const&, TArg const& arg)
        {
            return sycl::log(arg);
        }
    };

    //! The SYCL log2 trait specialization.
    template<typename TArg>
    struct Log2<math::Log2GenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::Log2GenericSycl const&, TArg const& arg)
        {
            return sycl::log2(arg);
        }
    };

    //! The SYCL log10 trait specialization.
    template<typename TArg>
    struct Log10<math::Log10GenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::Log10GenericSycl const&, TArg const& arg)
        {
            return sycl::log10(arg);
        }
    };

    //! The SYCL max trait specialization.
    template<typename Tx, typename Ty>
    struct Max<math::MaxGenericSycl, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
    {
        using TCommon = std::common_type_t<Tx, Ty>;

        auto operator()(math::MaxGenericSycl const&, Tx const& x, Ty const& y)
        {
            if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                return sycl::max(static_cast<TCommon>(x), static_cast<TCommon>(y));
            else if constexpr(std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>)
                return sycl::fmax(static_cast<TCommon>(x), static_cast<TCommon>(y));
            else if constexpr(
                (std::is_floating_point_v<Tx> && std::is_integral_v<Ty>)
                || (std::is_integral_v<Tx> && std::is_floating_point_v<Ty>) )
                return sycl::fmax(static_cast<double>(x), static_cast<double>(y)); // mirror CUDA back-end
            else
                static_assert(!sizeof(Tx), "Unsupported data types");
        }
    };

    //! The SYCL min trait specialization.
    template<typename Tx, typename Ty>
    struct Min<math::MinGenericSycl, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
    {
        auto operator()(math::MinGenericSycl const&, Tx const& x, Ty const& y)
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
                static_assert(!sizeof(Tx), "Unsupported data types");
        }
    };

    //! The SYCL pow trait specialization.
    template<typename TBase, typename TExp>
    struct Pow<
        math::PowGenericSycl,
        TBase,
        TExp,
        std::enable_if_t<std::is_floating_point_v<TBase> && std::is_floating_point_v<TExp>>>
    {
        using TCommon = std::common_type_t<TBase, TExp>;

        auto operator()(math::PowGenericSycl const&, TBase const& base, TExp const& exp)
        {
            return sycl::pow(static_cast<TCommon>(base), static_cast<TCommon>(exp));
        }
    };

    //! The SYCL remainder trait specialization.
    template<typename Tx, typename Ty>
    struct Remainder<
        math::RemainderGenericSycl,
        Tx,
        Ty,
        std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
    {
        using TCommon = std::common_type_t<Tx, Ty>;

        auto operator()(math::RemainderGenericSycl const&, Tx const& x, Ty const& y)
        {
            return sycl::remainder(static_cast<TCommon>(x), static_cast<TCommon>(y));
        }
    };

    //! The SYCL round trait specialization.
    template<typename TArg>
    struct Round<math::RoundGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::RoundGenericSycl const&, TArg const& arg)
        {
            return sycl::round(arg);
        }
    };

    //! The SYCL lround trait specialization.
    template<typename TArg>
    struct Lround<math::RoundGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::RoundGenericSycl const&, TArg const& arg)
        {
            return static_cast<long>(sycl::round(arg));
        }
    };

    //! The SYCL llround trait specialization.
    template<typename TArg>
    struct Llround<math::RoundGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::RoundGenericSycl const&, TArg const& arg)
        {
            return static_cast<long long>(sycl::round(arg));
        }
    };

    //! The SYCL rsqrt trait specialization.
    template<typename TArg>
    struct Rsqrt<math::RsqrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
    {
        auto operator()(math::RsqrtGenericSycl const&, TArg const& arg)
        {
            if constexpr(std::is_floating_point_v<TArg>)
                return sycl::rsqrt(arg);
            else if constexpr(std::is_integral_v<TArg>)
                return sycl::rsqrt(static_cast<double>(arg)); // mirror CUDA back-end and use double for ints
            else
                static_assert(!sizeof(TArg), "Unsupported data type");
        }
    };

    //! The SYCL sin trait specialization.
    template<typename TArg>
    struct Sin<math::SinGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::SinGenericSycl const&, TArg const& arg)
        {
            return sycl::sin(arg);
        }
    };

    //! The SYCL sinh trait specialization.
    template<typename TArg>
    struct Sinh<math::SinhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::SinhGenericSycl const&, TArg const& arg)
        {
            return sycl::sinh(arg);
        }
    };

    //! The SYCL sincos trait specialization.
    template<typename TArg>
    struct SinCos<math::SinCosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::SinCosGenericSycl const&, TArg const& arg, TArg& result_sin, TArg& result_cos) -> void
        {
            result_sin = sycl::sincos(arg, &result_cos);
        }
    };

    //! The SYCL sqrt trait specialization.
    template<typename TArg>
    struct Sqrt<math::SqrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
    {
        auto operator()(math::SqrtGenericSycl const&, TArg const& arg)
        {
            if constexpr(std::is_floating_point_v<TArg>)
                return sycl::sqrt(arg);
            else if constexpr(std::is_integral_v<TArg>)
                return sycl::sqrt(static_cast<double>(arg)); // mirror CUDA back-end and use double for ints
        }
    };

    //! The SYCL tan trait specialization.
    template<typename TArg>
    struct Tan<math::TanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::TanGenericSycl const&, TArg const& arg)
        {
            return sycl::tan(arg);
        }
    };

    //! The SYCL tanh trait specialization.
    template<typename TArg>
    struct Tanh<math::TanhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::TanhGenericSycl const&, TArg const& arg)
        {
            return sycl::tanh(arg);
        }
    };

    //! The SYCL trunc trait specialization.
    template<typename TArg>
    struct Trunc<math::TruncGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
    {
        auto operator()(math::TruncGenericSycl const&, TArg const& arg)
        {
            return sycl::trunc(arg);
        }
    };
} // namespace alpaka::math::trait

#endif
