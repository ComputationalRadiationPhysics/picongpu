/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Jan Stephan, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Defines.hpp"

#include <alpaka/alpaka.hpp>

#include <complex>
#include <type_traits>

namespace mathtest
{
    //! Helper trait to define a type conversion before passing T to a std:: math function
    //!
    //! The default implementation does no conversion
    //!
    //! @tparam T input type
    template<typename T>
    struct StdLibType
    {
        using type = T;
    };

    //! Specialization converting alpaka::Complex<T> to std::complex<T>
    template<typename T>
    struct StdLibType<alpaka::Complex<T>>
    {
        using type = std::complex<T>;
    };

// Can be used with operator() that will use either the std. function or the
// equivalent alpaka function (if an accelerator is passed additionally).
//! @param NAME The Name used for the Functor, e.g. OpAbs
//! @param ARITY Enum-type can be one ... n
//! @param STD_OP Function used for the host side, e.g. std::abs
//! @param ALPAKA_OP Function used for the device side, e.g. alpaka::math::abs.
//! @param ... List of Ranges. Needs to match the arity.
#define ALPAKA_TEST_MATH_OP_FUNCTOR(NAME, ARITY, STD_OP, ALPAKA_OP, ...)                                              \
    struct NAME                                                                                                       \
    {                                                                                                                 \
        /* ranges is not a constexpr, so that it's accessible via for loop*/                                          \
        static constexpr Arity arity = ARITY;                                                                         \
        static constexpr size_t arity_nr = static_cast<size_t>(ARITY);                                                \
        Range ranges[arity_nr] = {__VA_ARGS__};                                                                       \
                                                                                                                      \
        ALPAKA_NO_HOST_ACC_WARNING                                                                                    \
        template<                                                                                                     \
            typename TAcc,                                                                                            \
            typename... TArgs, /* SFINAE: Enables if called from device. */                                           \
            typename std::enable_if_t<!std::is_same_v<TAcc, std::nullptr_t>, int> = 0>                                \
        ALPAKA_FN_ACC auto execute(TAcc const& acc, TArgs const&... args) const                                       \
        {                                                                                                             \
            return ALPAKA_OP(acc, args...);                                                                           \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_NO_HOST_ACC_WARNING                                                                                    \
        template<                                                                                                     \
            typename TAcc = std::nullptr_t,                                                                           \
            typename TArg1, /* SFINAE: Enables if called from host. */                                                \
            typename std::enable_if_t<std::is_same_v<TAcc, std::nullptr_t>, int> = 0>                                 \
        ALPAKA_FN_HOST auto execute(TAcc const& /* acc */, TArg1 const& arg1) const                                   \
        {                                                                                                             \
            return STD_OP(typename StdLibType<TArg1>::type(arg1));                                                    \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_NO_HOST_ACC_WARNING                                                                                    \
        template<                                                                                                     \
            typename TAcc = std::nullptr_t,                                                                           \
            typename TArg1,                                                                                           \
            typename TArg2, /* SFINAE: Enables if called from host. */                                                \
            typename std::enable_if_t<std::is_same_v<TAcc, std::nullptr_t>, int> = 0>                                 \
        ALPAKA_FN_HOST auto execute(TAcc const& /* acc */, TArg1 const& arg1, TArg2 const& arg2) const                \
        {                                                                                                             \
            return STD_OP(typename StdLibType<TArg1>::type(arg1), typename StdLibType<TArg2>::type(arg2));            \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_NO_HOST_ACC_WARNING                                                                                    \
        template<                                                                                                     \
            typename TAcc = std::nullptr_t,                                                                           \
            typename TArg1,                                                                                           \
            typename TArg2,                                                                                           \
            typename TArg3, /* SFINAE: Enables if called from host. */                                                \
            typename std::enable_if_t<std::is_same_v<TAcc, std::nullptr_t>, int> = 0>                                 \
        ALPAKA_FN_HOST auto execute(TAcc const& /* acc */, TArg1 const& arg1, TArg2 const& arg2, TArg3 const& arg3)   \
            const                                                                                                     \
        {                                                                                                             \
            return STD_OP(                                                                                            \
                typename StdLibType<TArg1>::type(arg1),                                                               \
                typename StdLibType<TArg2>::type(arg2),                                                               \
                typename StdLibType<TArg3>::type(arg3));                                                              \
        }                                                                                                             \
                                                                                                                      \
        /* assigns args by arity */                                                                                   \
        ALPAKA_NO_HOST_ACC_WARNING                                                                                    \
        template<typename T, typename TAcc = std::nullptr_t>                                                          \
        ALPAKA_FN_HOST_ACC auto operator()(ArgsItem<T, Arity::Unary> const& args, TAcc const& acc = nullptr) const    \
        {                                                                                                             \
            return execute(acc, args.arg[0]);                                                                         \
        }                                                                                                             \
                                                                                                                      \
        /* assigns args by arity */                                                                                   \
        ALPAKA_NO_HOST_ACC_WARNING                                                                                    \
        template<typename T, typename TAcc = std::nullptr_t>                                                          \
        ALPAKA_FN_HOST_ACC auto operator()(ArgsItem<T, Arity::Binary> const& args, TAcc const& acc = nullptr) const   \
        {                                                                                                             \
            return execute(acc, args.arg[0], args.arg[1]);                                                            \
        }                                                                                                             \
                                                                                                                      \
        /* assigns args by arity */                                                                                   \
        ALPAKA_NO_HOST_ACC_WARNING                                                                                    \
        template<typename T, typename TAcc = std::nullptr_t>                                                          \
        ALPAKA_FN_HOST_ACC auto operator()(ArgsItem<T, Arity::Ternary> const& args, TAcc const& acc = nullptr) const  \
        {                                                                                                             \
            return execute(acc, args.arg[0], args.arg[1], args.arg[2]);                                               \
        }                                                                                                             \
                                                                                                                      \
        friend std::ostream& operator<<(std::ostream& out, const NAME& /* op */)                                      \
        {                                                                                                             \
            out << #NAME;                                                                                             \
            return out;                                                                                               \
        }                                                                                                             \
    };


    ALPAKA_TEST_MATH_OP_FUNCTOR(OpAbs, Arity::Unary, std::abs, alpaka::math::abs, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpAcos, Arity::Unary, std::acos, alpaka::math::acos, Range::OneNeighbourhood)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpAcosh, Arity::Unary, std::acosh, alpaka::math::acosh, Range::PositiveOnly)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpArg, Arity::Unary, std::arg, alpaka::math::arg, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpAsin, Arity::Unary, std::asin, alpaka::math::asin, Range::OneNeighbourhood)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpAsinh, Arity::Unary, std::asinh, alpaka::math::asinh, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpAtan, Arity::Unary, std::atan, alpaka::math::atan, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpAtanh, Arity::Unary, std::atanh, alpaka::math::atanh, Range::OneNeighbourhood)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpCbrt, Arity::Unary, std::cbrt, alpaka::math::cbrt, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpCeil, Arity::Unary, std::ceil, alpaka::math::ceil, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpCos, Arity::Unary, std::cos, alpaka::math::cos, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpCosh, Arity::Unary, std::cosh, alpaka::math::cosh, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpErf, Arity::Unary, std::erf, alpaka::math::erf, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpExp, Arity::Unary, std::exp, alpaka::math::exp, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpFloor, Arity::Unary, std::floor, alpaka::math::floor, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpLog, Arity::Unary, std::log, alpaka::math::log, Range::PositiveOnly)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpLog2, Arity::Unary, std::log2, alpaka::math::log2, Range::PositiveOnly)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpLog10, Arity::Unary, std::log10, alpaka::math::log10, Range::PositiveOnly)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpRound, Arity::Unary, std::round, alpaka::math::round, Range::Unrestricted)

    // There is no std implementation, look in Defines.hpp.
    ALPAKA_TEST_MATH_OP_FUNCTOR(OpRsqrt, Arity::Unary, rsqrt, alpaka::math::rsqrt, Range::PositiveOnly)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpSin, Arity::Unary, std::sin, alpaka::math::sin, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpSinh, Arity::Unary, std::sinh, alpaka::math::sinh, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpSqrt, Arity::Unary, std::sqrt, alpaka::math::sqrt, Range::PositiveAndZero)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpTan, Arity::Unary, std::tan, alpaka::math::tan, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpTanh, Arity::Unary, std::tanh, alpaka::math::tanh, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpTrunc, Arity::Unary, std::trunc, alpaka::math::trunc, Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpIsnan, Arity::Unary, std::isnan, alpaka::math::isnan, Range::Anything)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpIsinf, Arity::Unary, std::isinf, alpaka::math::isinf, Range::Anything)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpIsfinite, Arity::Unary, std::isfinite, alpaka::math::isfinite, Range::Anything)

    // All binary operators.
    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpAtan2,
        Arity::Binary,
        std::atan2,
        alpaka::math::atan2,
        Range::NotZero,
        Range::NotZero)

    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpCopysign,
        Arity::Binary,
        std::copysign,
        alpaka::math::copysign,
        Range::Unrestricted,
        Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpFmod,
        Arity::Binary,
        std::fmod,
        alpaka::math::fmod,
        Range::Unrestricted,
        Range::NotZero)

    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpMax,
        Arity::Binary,
        std::max,
        alpaka::math::max,
        Range::Unrestricted,
        Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpMin,
        Arity::Binary,
        std::min,
        alpaka::math::min,
        Range::Unrestricted,
        Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpPow,
        Arity::Binary,
        std::pow,
        alpaka::math::pow,
        Range::PositiveAndZero,
        Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpRemainder,
        Arity::Binary,
        std::remainder,
        alpaka::math::remainder,
        Range::Unrestricted,
        Range::NotZero)

    // All ternary operators.
    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpFma,
        Arity::Ternary,
        std::fma,
        alpaka::math::fma,
        Range::Unrestricted,
        Range::Unrestricted,
        Range::Unrestricted)

    // Unary functors to be used only for real types
    using UnaryFunctorsReal = std::tuple<
        OpAbs,
        OpAcos,
        OpAcosh,
        OpArg,
        OpAsin,
        OpAsinh,
        OpAtan,
        OpAtanh,
        OpCbrt,
        OpCeil,
        OpCos,
        OpCosh,
        OpErf,
        OpExp,
        OpFloor,
        OpLog,
        OpLog2,
        OpLog10,
        OpRound,
        OpRsqrt,
        OpSin,
        OpSinh,
        OpSqrt,
        OpTan,
        OpTanh,
        OpTrunc,
        OpIsnan,
        OpIsinf,
        OpIsfinite>;

    // Binary functors to be used only for real types
    using BinaryFunctorsReal = std::tuple<OpAtan2, OpCopysign, OpFmod, OpMax, OpMin, OpPow, OpRemainder>;

    // Ternary functors to be used only for real types
    using TernaryFunctorsReal = std::tuple<OpFma>;

    // For complex numbers also test arithmetic operations
    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpDivides,
        Arity::Binary,
        std::divides<>{},
        divides,
        Range::Unrestricted,
        Range::NotZero)

    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpMinus,
        Arity::Binary,
        std::minus<>{},
        minus,
        Range::Unrestricted,
        Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpMultiplies,
        Arity::Binary,
        std::multiplies<>{},
        multiplies,
        Range::Unrestricted,
        Range::Unrestricted)

    ALPAKA_TEST_MATH_OP_FUNCTOR(OpPlus, Arity::Binary, std::plus<>{}, plus, Range::Unrestricted, Range::Unrestricted)

    // conj() is only tested for complex as it returns a complex type for real arguments
    // and it doesn't fit the existing tests' infrastructure
    ALPAKA_TEST_MATH_OP_FUNCTOR(OpConj, Arity::Unary, std::conj, alpaka::math::conj, Range::Unrestricted)

    // As a workaround for complex 0^0 unit tests issues, test complex pow for positive range only
    ALPAKA_TEST_MATH_OP_FUNCTOR(
        OpPowComplex,
        Arity::Binary,
        std::pow,
        alpaka::math::pow,
        Range::PositiveOnly,
        Range::Unrestricted)

    // Unary functors to be used for both real and complex types
    using UnaryFunctorsComplex = std::tuple<
        OpAbs,
        OpAcos,
        OpAcosh,
        OpArg,
        OpAsin,
        OpAsinh,
        OpAtan,
        OpAtanh,
        OpConj,
        OpCos,
        OpCosh,
        OpExp,
        OpLog,
        OpLog10,
        OpRsqrt,
        OpSin,
        OpSinh,
        OpSqrt,
        OpTan,
        OpTanh>;

    // Binary functors to be used for complex types
    using BinaryFunctorsComplex = std::tuple<OpDivides, OpMinus, OpMultiplies, OpPlus, OpPowComplex>;
} // namespace mathtest
