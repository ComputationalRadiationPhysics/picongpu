/* Copyright 2023 Axel Huebl, Benjamin Worpitz, Matthias Werner, Bert Wesarg, Valentin Gehrke, René Widera,
 * Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/CudaHipCommon.hpp"
#include "alpaka/core/Decay.hpp"
#include "alpaka/core/UniformCudaHip.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/math/Complex.hpp"
#include "alpaka/math/Traits.hpp"

#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::math
{
    //! The CUDA built in abs.
    class AbsUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAbs, AbsUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in acos.
    class AcosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAcos, AcosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in acosh.
    class AcoshUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAcosh, AcoshUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in arg.
    class ArgUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathArg, ArgUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in asin.
    class AsinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAsin, AsinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in asinh.
    class AsinhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAsinh, AsinhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atan.
    class AtanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan, AtanUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atanh.
    class AtanhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtanh, AtanhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in atan2.
    class Atan2UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan2, Atan2UniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cbrt.
    class CbrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCbrt, CbrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in ceil.
    class CeilUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCeil, CeilUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in conj.
    class ConjUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathConj, ConjUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in copysign.
    class CopysignUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptMathCopysign, CopysignUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cos.
    class CosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCos, CosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in cosh.
    class CoshUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCosh, CoshUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in erf.
    class ErfUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathErf, ErfUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in exp.
    class ExpUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathExp, ExpUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in floor.
    class FloorUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFloor, FloorUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in fma.
    class FmaUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFma, FmaUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in fmod.
    class FmodUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFmod, FmodUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isfinite.
    class IsfiniteUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptMathIsfinite, IsfiniteUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isinf.
    class IsinfUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathIsinf, IsinfUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in isnan.
    class IsnanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathIsnan, IsnanUniformCudaHipBuiltIn>
    {
    };

    // ! The CUDA built in log.
    class LogUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathLog, LogUniformCudaHipBuiltIn>
    {
    };

    // ! The CUDA built in log2.
    class Log2UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathLog2, Log2UniformCudaHipBuiltIn>
    {
    };

    // ! The CUDA built in log10.
    class Log10UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathLog10, Log10UniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in max.
    class MaxUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMax, MaxUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in min.
    class MinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMin, MinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in pow.
    class PowUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathPow, PowUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA built in remainder.
    class RemainderUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptMathRemainder, RemainderUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA round.
    class RoundUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRound, RoundUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA rsqrt.
    class RsqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRsqrt, RsqrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sin.
    class SinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSin, SinUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sinh.
    class SinhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSinh, SinhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sincos.
    class SinCosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSinCos, SinCosUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA sqrt.
    class SqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSqrt, SqrtUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA tan.
    class TanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTan, TanUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA tanh.
    class TanhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTanh, TanhUniformCudaHipBuiltIn>
    {
    };

    //! The CUDA trunc.
    class TruncUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTrunc, TruncUniformCudaHipBuiltIn>
    {
    };

    //! The standard library math trait specializations.
    class MathUniformCudaHipBuiltIn
        : public AbsUniformCudaHipBuiltIn
        , public AcosUniformCudaHipBuiltIn
        , public AcoshUniformCudaHipBuiltIn
        , public ArgUniformCudaHipBuiltIn
        , public AsinUniformCudaHipBuiltIn
        , public AsinhUniformCudaHipBuiltIn
        , public AtanUniformCudaHipBuiltIn
        , public AtanhUniformCudaHipBuiltIn
        , public Atan2UniformCudaHipBuiltIn
        , public CbrtUniformCudaHipBuiltIn
        , public CeilUniformCudaHipBuiltIn
        , public ConjUniformCudaHipBuiltIn
        , public CopysignUniformCudaHipBuiltIn
        , public CosUniformCudaHipBuiltIn
        , public CoshUniformCudaHipBuiltIn
        , public ErfUniformCudaHipBuiltIn
        , public ExpUniformCudaHipBuiltIn
        , public FloorUniformCudaHipBuiltIn
        , public FmaUniformCudaHipBuiltIn
        , public FmodUniformCudaHipBuiltIn
        , public LogUniformCudaHipBuiltIn
        , public Log2UniformCudaHipBuiltIn
        , public Log10UniformCudaHipBuiltIn
        , public MaxUniformCudaHipBuiltIn
        , public MinUniformCudaHipBuiltIn
        , public PowUniformCudaHipBuiltIn
        , public RemainderUniformCudaHipBuiltIn
        , public RoundUniformCudaHipBuiltIn
        , public RsqrtUniformCudaHipBuiltIn
        , public SinUniformCudaHipBuiltIn
        , public SinhUniformCudaHipBuiltIn
        , public SinCosUniformCudaHipBuiltIn
        , public SqrtUniformCudaHipBuiltIn
        , public TanUniformCudaHipBuiltIn
        , public TanhUniformCudaHipBuiltIn
        , public TruncUniformCudaHipBuiltIn
        , public IsnanUniformCudaHipBuiltIn
        , public IsinfUniformCudaHipBuiltIn
        , public IsfiniteUniformCudaHipBuiltIn
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)
#            include <cuda_runtime.h>
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__)
#            include <hip/math_functions.h>
#        endif

    namespace trait
    {
        //! The CUDA abs trait specialization for real types.
        template<typename TArg>
        struct Abs<AbsUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_signed_v<TArg>>>
        {
            __host__ __device__ auto operator()(AbsUniformCudaHipBuiltIn const& /* abs_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::fabsf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::fabs(arg);
                else if constexpr(is_decayed_v<TArg, int>)
                    return ::abs(arg);
                else if constexpr(is_decayed_v<TArg, long int>)
                    return ::labs(arg);
                else if constexpr(is_decayed_v<TArg, long long int>)
                    return ::llabs(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA abs trait specialization for complex types.
        template<typename T>
        struct Abs<AbsUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                return sqrt(ctx, arg.real() * arg.real() + arg.imag() * arg.imag());
            }
        };

        //! The CUDA acos trait specialization for real types.
        template<typename TArg>
        struct Acos<AcosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(AcosUniformCudaHipBuiltIn const& /* acos_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::acosf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::acos(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA acos trait specialization for complex types.
        template<typename T>
        struct Acos<AcosUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // This holds everywhere, including the branch cuts: acos(z) = -i * ln(z + i * sqrt(1 - z^2))
                return Complex<T>{0.0, -1.0} * log(ctx, arg + Complex<T>{0.0, 1.0} * sqrt(ctx, T(1.0) - arg * arg));
            }
        };

        //! The CUDA acosh trait specialization for real types.
        template<typename TArg>
        struct Acosh<AcoshUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(AcoshUniformCudaHipBuiltIn const& /* acosh_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::acoshf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::acosh(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA acosh trait specialization for complex types.
        template<typename T>
        struct Acosh<AcoshUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // acos(z) = ln(z + sqrt(z-1) * sqrt(z+1))
                return log(ctx, arg + sqrt(ctx, arg - static_cast<T>(1.0)) * sqrt(ctx, arg + static_cast<T>(1.0)));
            }
        };

        //! The CUDA arg trait specialization for real types.
        template<typename TArgument>
        struct Arg<ArgUniformCudaHipBuiltIn, TArgument, std::enable_if_t<std::is_floating_point_v<TArgument>>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, TArgument const& argument)
            {
                // Fall back to atan2 so that boundary cases are resolved consistently
                return atan2(ctx, TArgument{0.0}, argument);
            }
        };

        //! The CUDA arg Complex<T> specialization for complex types.
        template<typename T>
        struct Arg<ArgUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& argument)
            {
                return atan2(ctx, argument.imag(), argument.real());
            }
        };

        //! The CUDA asin trait specialization for real types.
        template<typename TArg>
        struct Asin<AsinUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(AsinUniformCudaHipBuiltIn const& /* asin_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::asinf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::asin(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA asin trait specialization for complex types.
        template<typename T>
        struct Asin<AsinUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // This holds everywhere, including the branch cuts: asin(z) = i * ln(sqrt(1 - z^2) - i * z)
                return Complex<T>{0.0, 1.0} * log(ctx, sqrt(ctx, T(1.0) - arg * arg) - Complex<T>{0.0, 1.0} * arg);
            }
        };

        //! The CUDA asinh trait specialization for real types.
        template<typename TArg>
        struct Asinh<AsinhUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(AsinhUniformCudaHipBuiltIn const& /* asinh_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::asinhf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::asinh(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA asinh trait specialization for complex types.
        template<typename T>
        struct Asinh<AsinhUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // asinh(z) = ln(z + sqrt(z^2 + 1))
                return log(ctx, arg + sqrt(ctx, arg * arg + static_cast<T>(1.0)));
            }
        };

        //! The CUDA atan trait specialization for real types.
        template<typename TArg>
        struct Atan<AtanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(AtanUniformCudaHipBuiltIn const& /* atan_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::atanf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::atan(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA atan trait specialization for complex types.
        template<typename T>
        struct Atan<AtanUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // This holds everywhere, including the branch cuts: atan(z) = -i/2 * ln((i - z) / (i + z))
                return Complex<T>{0.0, -0.5} * log(ctx, (Complex<T>{0.0, 1.0} - arg) / (Complex<T>{0.0, 1.0} + arg));
            }
        };

        //! The CUDA atanh trait specialization for real types.
        template<typename TArg>
        struct Atanh<AtanhUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(AtanhUniformCudaHipBuiltIn const& /* atanh_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::atanhf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::atanh(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA atanh trait specialization for complex types.
        template<typename T>
        struct Atanh<AtanhUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                //  atanh(z) = 0.5 * (ln(1 + z) - ln(1 - z))
                return static_cast<T>(0.5)
                       * (log(ctx, static_cast<T>(1.0) + arg) - log(ctx, static_cast<T>(1.0) - arg));
            }
        };

        //! The CUDA atan2 trait specialization.
        template<typename Ty, typename Tx>
        struct Atan2<
            Atan2UniformCudaHipBuiltIn,
            Ty,
            Tx,
            std::enable_if_t<std::is_floating_point_v<Ty> && std::is_floating_point_v<Tx>>>
        {
            __host__ __device__ auto operator()(
                Atan2UniformCudaHipBuiltIn const& /* atan2_ctx */,
                Ty const& y,
                Tx const& x)
            {
                if constexpr(is_decayed_v<Ty, float> && is_decayed_v<Tx, float>)
                    return ::atan2f(y, x);
                else if constexpr(is_decayed_v<Ty, double> || is_decayed_v<Tx, double>)
                    return ::atan2(y, x);
                else
                    static_assert(!sizeof(Ty), "Unsupported data type");

                ALPAKA_UNREACHABLE(Ty{});
            }
        };

        //! The CUDA cbrt trait specialization.
        template<typename TArg>
        struct Cbrt<CbrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            __host__ __device__ auto operator()(CbrtUniformCudaHipBuiltIn const& /* cbrt_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::cbrtf(arg);
                else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
                    return ::cbrt(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA ceil trait specialization.
        template<typename TArg>
        struct Ceil<CeilUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(CeilUniformCudaHipBuiltIn const& /* ceil_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::ceilf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::ceil(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA conj trait specialization for real types.
        template<typename TArg>
        struct Conj<ConjUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(ConjUniformCudaHipBuiltIn const& /* conj_ctx */, TArg const& arg)
            {
                return Complex<TArg>{arg, TArg{0.0}};
            }
        };

        //! The CUDA conj specialization for complex types.
        template<typename T>
        struct Conj<ConjUniformCudaHipBuiltIn, Complex<T>>
        {
            __host__ __device__ auto operator()(ConjUniformCudaHipBuiltIn const& /* conj_ctx */, Complex<T> const& arg)
            {
                return Complex<T>{arg.real(), -arg.imag()};
            }
        };

        //! The CUDA copysign trait specialization for real types.
        template<typename TMag, typename TSgn>
        struct Copysign<
            CopysignUniformCudaHipBuiltIn,
            TMag,
            TSgn,
            std::enable_if_t<std::is_floating_point_v<TMag> && std::is_floating_point_v<TSgn>>>
        {
            __host__ __device__ auto operator()(
                CopysignUniformCudaHipBuiltIn const& /* copysign_ctx */,
                TMag const& mag,
                TSgn const& sgn)
            {
                if constexpr(is_decayed_v<TMag, float> && is_decayed_v<TSgn, float>)
                    return ::copysignf(mag, sgn);
                else if constexpr(is_decayed_v<TMag, double> || is_decayed_v<TSgn, double>)
                    return ::copysign(mag, sgn);
                else
                    static_assert(!sizeof(TMag), "Unsupported data type");

                ALPAKA_UNREACHABLE(TMag{});
            }
        };

        //! The CUDA cos trait specialization for real types.
        template<typename TArg>
        struct Cos<CosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(CosUniformCudaHipBuiltIn const& /* cos_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::cosf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::cos(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA cos trait specialization for complex types.
        template<typename T>
        struct Cos<CosUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // cos(z) = 0.5 * (exp(i * z) + exp(-i * z))
                return T(0.5) * (exp(ctx, Complex<T>{0.0, 1.0} * arg) + exp(ctx, Complex<T>{0.0, -1.0} * arg));
            }
        };

        //! The CUDA cosh trait specialization for real types.
        template<typename TArg>
        struct Cosh<CoshUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(CoshUniformCudaHipBuiltIn const& /* cos_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::coshf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::cosh(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA cosh trait specialization for complex types.
        template<typename T>
        struct Cosh<CoshUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // cosh(z) = 0.5 * (exp(z) + exp(-z))
                return T(0.5) * (exp(ctx, arg) + exp(ctx, static_cast<T>(-1.0) * arg));
            }
        };

        //! The CUDA erf trait specialization.
        template<typename TArg>
        struct Erf<ErfUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(ErfUniformCudaHipBuiltIn const& /* erf_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::erff(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::erf(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA exp trait specialization for real types.
        template<typename TArg>
        struct Exp<ExpUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(ExpUniformCudaHipBuiltIn const& /* exp_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::expf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::exp(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA exp trait specialization for complex types.
        template<typename T>
        struct Exp<ExpUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // exp(z) = exp(x + iy) = exp(x) * (cos(y) + i * sin(y))
                auto re = T{}, im = T{};
                sincos(ctx, arg.imag(), im, re);
                return exp(ctx, arg.real()) * Complex<T>{re, im};
            }
        };

        //! The CUDA floor trait specialization.
        template<typename TArg>
        struct Floor<FloorUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(FloorUniformCudaHipBuiltIn const& /* floor_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::floorf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::floor(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA fma trait specialization.
        template<typename Tx, typename Ty, typename Tz>
        struct Fma<
            FmaUniformCudaHipBuiltIn,
            Tx,
            Ty,
            Tz,
            std::enable_if_t<
                std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty> && std::is_floating_point_v<Tz>>>
        {
            __host__ __device__ auto operator()(
                FmaUniformCudaHipBuiltIn const& /* fma_ctx */,
                Tx const& x,
                Ty const& y,
                Tz const& z)
            {
                if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float> && is_decayed_v<Tz, float>)
                    return ::fmaf(x, y, z);
                else if constexpr(is_decayed_v<Tx, double> || is_decayed_v<Ty, double> || is_decayed_v<Tz, double>)
                    return ::fma(x, y, z);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                using Ret [[maybe_unused]] = std::conditional_t<
                    is_decayed_v<Tx, float> && is_decayed_v<Ty, float> && is_decayed_v<Tz, float>,
                    float,
                    double>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA fmod trait specialization.
        template<typename Tx, typename Ty>
        struct Fmod<
            FmodUniformCudaHipBuiltIn,
            Tx,
            Ty,
            std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
        {
            __host__ __device__ auto operator()(
                FmodUniformCudaHipBuiltIn const& /* fmod_ctx */,
                Tx const& x,
                Ty const& y)
            {
                if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                    return ::fmodf(x, y);
                else if constexpr(is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
                    return ::fmod(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                using Ret [[maybe_unused]]
                = std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA isfinite trait specialization.
        template<typename TArg>
        struct Isfinite<IsfiniteUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(IsfiniteUniformCudaHipBuiltIn const& /* ctx */, TArg const& arg)
            {
                return ::isfinite(arg);
            }
        };

        //! The CUDA isinf trait specialization.
        template<typename TArg>
        struct Isinf<IsinfUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(IsinfUniformCudaHipBuiltIn const& /* ctx */, TArg const& arg)
            {
                return ::isinf(arg);
            }
        };

        //! The CUDA isnan trait specialization.
        template<typename TArg>
        struct Isnan<IsnanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(IsnanUniformCudaHipBuiltIn const& /* ctx */, TArg const& arg)
            {
                return ::isnan(arg);
            }
        };

        //! The CUDA log trait specialization for real types.
        template<typename TArg>
        struct Log<LogUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(LogUniformCudaHipBuiltIn const& /* log_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::logf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::log(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA log trait specialization for complex types.
        template<typename T>
        struct Log<LogUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& argument)
            {
                // Branch cut along the negative real axis (same as for std::complex),
                // principal value of ln(z) = ln(|z|) + i * arg(z)
                return log(ctx, abs(ctx, argument)) + Complex<T>{0.0, 1.0} * arg(ctx, argument);
            }
        };

        //! The CUDA log2 trait specialization for real types.
        template<typename TArg>
        struct Log2<Log2UniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(Log2UniformCudaHipBuiltIn const& /* log2_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::log2f(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::log2(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA log10 trait specialization for real types.
        template<typename TArg>
        struct Log10<Log10UniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(Log10UniformCudaHipBuiltIn const& /* log10_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::log10f(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::log10(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA log10 trait specialization for complex types.
        template<typename T>
        struct Log10<Log10UniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& argument)
            {
                return log(ctx, argument) / log(ctx, static_cast<T>(10));
            }
        };

        //! The CUDA max trait specialization.
        template<typename Tx, typename Ty>
        struct Max<
            MaxUniformCudaHipBuiltIn,
            Tx,
            Ty,
            std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            __host__ __device__ auto operator()(
                MaxUniformCudaHipBuiltIn const& /* max_ctx */,
                Tx const& x,
                Ty const& y)
            {
                if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                    return ::max(x, y);
                else if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                    return ::fmaxf(x, y);
                else if constexpr(
                    is_decayed_v<Tx, double> || is_decayed_v<Ty, double>
                    || (is_decayed_v<Tx, float> && std::is_integral_v<Ty>)
                    || (std::is_integral_v<Tx> && is_decayed_v<Ty, float>) )
                    return ::fmax(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                using Ret [[maybe_unused]] = std::conditional_t<
                    std::is_integral_v<Tx> && std::is_integral_v<Ty>,
                    decltype(::max(x, y)),
                    std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA min trait specialization.
        template<typename Tx, typename Ty>
        struct Min<
            MinUniformCudaHipBuiltIn,
            Tx,
            Ty,
            std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            __host__ __device__ auto operator()(
                MinUniformCudaHipBuiltIn const& /* min_ctx */,
                Tx const& x,
                Ty const& y)
            {
                if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                    return ::min(x, y);
                else if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                    return ::fminf(x, y);
                else if constexpr(
                    is_decayed_v<Tx, double> || is_decayed_v<Ty, double>
                    || (is_decayed_v<Tx, float> && std::is_integral_v<Ty>)
                    || (std::is_integral_v<Tx> && is_decayed_v<Ty, float>) )
                    return ::fmin(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                using Ret [[maybe_unused]] = std::conditional_t<
                    std::is_integral_v<Tx> && std::is_integral_v<Ty>,
                    decltype(::min(x, y)),
                    std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA pow trait specialization for real types.
        template<typename TBase, typename TExp>
        struct Pow<
            PowUniformCudaHipBuiltIn,
            TBase,
            TExp,
            std::enable_if_t<std::is_floating_point_v<TBase> && std::is_floating_point_v<TExp>>>
        {
            __host__ __device__ auto operator()(
                PowUniformCudaHipBuiltIn const& /* pow_ctx */,
                TBase const& base,
                TExp const& exp)
            {
                if constexpr(is_decayed_v<TBase, float> && is_decayed_v<TExp, float>)
                    return ::powf(base, exp);
                else if constexpr(is_decayed_v<TBase, double> || is_decayed_v<TExp, double>)
                    return ::pow(static_cast<double>(base), static_cast<double>(exp));
                else
                    static_assert(!sizeof(TBase), "Unsupported data type");

                using Ret [[maybe_unused]]
                = std::conditional_t<is_decayed_v<TBase, float> && is_decayed_v<TExp, float>, float, double>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA pow trait specialization for complex types.
        template<typename T, typename U>
        struct Pow<PowUniformCudaHipBuiltIn, Complex<T>, Complex<U>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& base, Complex<U> const& exponent)
            {
                // Type promotion matching rules of complex std::pow but simplified given our math only supports float
                // and double, no long double.
                using Promoted
                    = Complex<std::conditional_t<is_decayed_v<T, float> && is_decayed_v<U, float>, float, double>>;
                // pow(z1, z2) = e^(z2 * log(z1))
                return exp(ctx, Promoted{exponent} * log(ctx, Promoted{base}));
            }
        };

        //! The CUDA pow trait specialization for complex and real types.
        template<typename T, typename U>
        struct Pow<PowUniformCudaHipBuiltIn, Complex<T>, U>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& base, U const& exponent)
            {
                return pow(ctx, base, Complex<U>{exponent});
            }
        };

        //! The CUDA pow trait specialization for real and complex types.
        template<typename T, typename U>
        struct Pow<PowUniformCudaHipBuiltIn, T, Complex<U>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, T const& base, Complex<U> const& exponent)
            {
                return pow(ctx, Complex<T>{base}, exponent);
            }
        };

        //! The CUDA remainder trait specialization.
        template<typename Tx, typename Ty>
        struct Remainder<
            RemainderUniformCudaHipBuiltIn,
            Tx,
            Ty,
            std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
        {
            __host__ __device__ auto operator()(
                RemainderUniformCudaHipBuiltIn const& /* remainder_ctx */,
                Tx const& x,
                Ty const& y)
            {
                if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                    return ::remainderf(x, y);
                else if constexpr(is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
                    return ::remainder(x, y);
                else
                    static_assert(!sizeof(Tx), "Unsupported data type");

                using Ret [[maybe_unused]]
                = std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA round trait specialization.
        template<typename TArg>
        struct Round<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(RoundUniformCudaHipBuiltIn const& /* round_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::roundf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::round(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA lround trait specialization.
        template<typename TArg>
        struct Lround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(RoundUniformCudaHipBuiltIn const& /* lround_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::lroundf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::lround(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(long{});
            }
        };

        //! The CUDA llround trait specialization.
        template<typename TArg>
        struct Llround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(RoundUniformCudaHipBuiltIn const& /* llround_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::llroundf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::llround(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                // NVCC versions before 11.3 are unable to compile 'long long{}': "type name is not allowed".
                using Ret [[maybe_unused]] = long long;
                ALPAKA_UNREACHABLE(Ret{});
            }
        };

        //! The CUDA rsqrt trait specialization for real types.
        template<typename TArg>
        struct Rsqrt<RsqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            __host__ __device__ auto operator()(RsqrtUniformCudaHipBuiltIn const& /* rsqrt_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::rsqrtf(arg);
                else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
                    return ::rsqrt(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA rsqrt trait specialization for complex types.
        template<typename T>
        struct Rsqrt<RsqrtUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                return T{1.0} / sqrt(ctx, arg);
            }
        };

        //! The CUDA sin trait specialization for real types.
        template<typename TArg>
        struct Sin<SinUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(SinUniformCudaHipBuiltIn const& /* sin_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::sinf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::sin(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA sin trait specialization for complex types.
        template<typename T>
        struct Sin<SinUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // sin(z) = (exp(i * z) - exp(-i * z)) / 2i
                return (exp(ctx, Complex<T>{0.0, 1.0} * arg) - exp(ctx, Complex<T>{0.0, -1.0} * arg))
                       / Complex<T>{0.0, 2.0};
            }
        };

        //! The CUDA sinh trait specialization for real types.
        template<typename TArg>
        struct Sinh<SinhUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(SinhUniformCudaHipBuiltIn const& /* sinh_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::sinhf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::sinh(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA sinh trait specialization for complex types.
        template<typename T>
        struct Sinh<SinhUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // sinh(z) = (exp(z) - exp(-i * z)) / 2
                return (exp(ctx, arg) - exp(ctx, static_cast<T>(-1.0) * arg)) / static_cast<T>(2.0);
            }
        };

        //! The CUDA sincos trait specialization for real types.
        template<typename TArg>
        struct SinCos<SinCosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(
                SinCosUniformCudaHipBuiltIn const& /* sincos_ctx */,
                TArg const& arg,
                TArg& result_sin,
                TArg& result_cos) -> void
            {
                if constexpr(is_decayed_v<TArg, float>)
                    ::sincosf(arg, &result_sin, &result_cos);
                else if constexpr(is_decayed_v<TArg, double>)
                    ::sincos(arg, &result_sin, &result_cos);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");
            }
        };

        //! The CUDA sincos trait specialization for complex types.
        template<typename T>
        struct SinCos<SinCosUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(
                TCtx const& ctx,
                Complex<T> const& arg,
                Complex<T>& result_sin,
                Complex<T>& result_cos) -> void
            {
                result_sin = sin(ctx, arg);
                result_cos = cos(ctx, arg);
            }
        };

        //! The CUDA sqrt trait specialization for real types.
        template<typename TArg>
        struct Sqrt<SqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            __host__ __device__ auto operator()(SqrtUniformCudaHipBuiltIn const& /* sqrt_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::sqrtf(arg);
                else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
                    return ::sqrt(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA sqrt trait specialization for complex types.
        template<typename T>
        struct Sqrt<SqrtUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& argument)
            {
                // Branch cut along the negative real axis (same as for std::complex),
                // principal value of sqrt(z) = sqrt(|z|) * e^(i * arg(z) / 2)
                auto const halfArg = T(0.5) * arg(ctx, argument);
                auto re = T{}, im = T{};
                sincos(ctx, halfArg, im, re);
                return sqrt(ctx, abs(ctx, argument)) * Complex<T>(re, im);
            }
        };

        //! The CUDA tan trait specialization for real types.
        template<typename TArg>
        struct Tan<TanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(TanUniformCudaHipBuiltIn const& /* tan_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::tanf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::tan(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA tan trait specialization for complex types.
        template<typename T>
        struct Tan<TanUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // tan(z) = i * (e^-iz - e^iz) / (e^-iz + e^iz) = i * (1 - e^2iz) / (1 + e^2iz)
                // Warning: this straightforward implementation can easily result in NaN as 0/0 or inf/inf.
                auto const expValue = exp(ctx, Complex<T>{0.0, 2.0} * arg);
                return Complex<T>{0.0, 1.0} * (T{1.0} - expValue) / (T{1.0} + expValue);
            }
        };

        //! The CUDA tanh trait specialization for real types.
        template<typename TArg>
        struct Tanh<TanhUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(TanhUniformCudaHipBuiltIn const& /* tanh_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::tanhf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::tanh(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };

        //! The CUDA tanh trait specialization for complex types.
        template<typename T>
        struct Tanh<TanhUniformCudaHipBuiltIn, Complex<T>>
        {
            //! Take context as original (accelerator) type, since we call other math functions
            template<typename TCtx>
            __host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
            {
                // tanh(z) = (e^z - e^-z)/(e^z+e^-z)
                return (exp(ctx, arg) - exp(ctx, static_cast<T>(-1.0) * arg))
                       / (exp(ctx, arg) + exp(ctx, static_cast<T>(-1.0) * arg));
            }
        };

        //! The CUDA trunc trait specialization.
        template<typename TArg>
        struct Trunc<TruncUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            __host__ __device__ auto operator()(TruncUniformCudaHipBuiltIn const& /* trunc_ctx */, TArg const& arg)
            {
                if constexpr(is_decayed_v<TArg, float>)
                    return ::truncf(arg);
                else if constexpr(is_decayed_v<TArg, double>)
                    return ::trunc(arg);
                else
                    static_assert(!sizeof(TArg), "Unsupported data type");

                ALPAKA_UNREACHABLE(TArg{});
            }
        };
    } // namespace trait
#    endif
} // namespace alpaka::math

#endif
