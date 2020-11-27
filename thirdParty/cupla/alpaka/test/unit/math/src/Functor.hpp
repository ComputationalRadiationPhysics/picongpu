/** Copyright 2019 Jakob Krude, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "Defines.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace alpaka
{
    namespace test
    {
        namespace unit
        {
            namespace math
            {
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
        const Range ranges[arity_nr] = {__VA_ARGS__};                                                                 \
                                                                                                                      \
        ALPAKA_NO_HOST_ACC_WARNING                                                                                    \
        template<                                                                                                     \
            typename TAcc,                                                                                            \
            typename... TArgs, /* SFINAE: Enables if called from device. */                                           \
            typename std::enable_if<!std::is_same<TAcc, std::nullptr_t>::value, int>::type = 0>                       \
        ALPAKA_FN_ACC auto execute(TAcc const& acc, TArgs const&... args) const                                       \
        {                                                                                                             \
            return ALPAKA_OP(acc, args...);                                                                           \
        }                                                                                                             \
                                                                                                                      \
        ALPAKA_NO_HOST_ACC_WARNING                                                                                    \
        template<                                                                                                     \
            typename TAcc = std::nullptr_t,                                                                           \
            typename... TArgs, /* SFINAE: Enables if called from host. */                                             \
            typename std::enable_if<std::is_same<TAcc, std::nullptr_t>::value, int>::type = 0>                        \
        ALPAKA_FN_HOST auto execute(TAcc const& acc, TArgs const&... args) const                                      \
        {                                                                                                             \
            alpaka::ignore_unused(acc);                                                                               \
            return STD_OP(args...);                                                                                   \
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
        friend std::ostream& operator<<(std::ostream& out, const NAME& op)                                            \
        {                                                                                                             \
            out << #NAME;                                                                                             \
            alpaka::ignore_unused(op);                                                                                \
            return out;                                                                                               \
        }                                                                                                             \
    };


                ALPAKA_TEST_MATH_OP_FUNCTOR(OpAbs, Arity::Unary, std::abs, alpaka::math::abs, Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(
                    OpAcos,
                    Arity::Unary,
                    std::acos,
                    alpaka::math::acos,
                    Range::OneNeighbourhood)

                ALPAKA_TEST_MATH_OP_FUNCTOR(
                    OpAsin,
                    Arity::Unary,
                    std::asin,
                    alpaka::math::asin,
                    Range::OneNeighbourhood)

                ALPAKA_TEST_MATH_OP_FUNCTOR(OpAtan, Arity::Unary, std::atan, alpaka::math::atan, Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(OpCbrt, Arity::Unary, std::cbrt, alpaka::math::cbrt, Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(OpCeil, Arity::Unary, std::ceil, alpaka::math::ceil, Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(OpCos, Arity::Unary, std::cos, alpaka::math::cos, Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(OpErf, Arity::Unary, std::erf, alpaka::math::erf, Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(OpExp, Arity::Unary, std::exp, alpaka::math::exp, Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(
                    OpFloor,
                    Arity::Unary,
                    std::floor,
                    alpaka::math::floor,
                    Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(OpLog, Arity::Unary, std::log, alpaka::math::log, Range::PositiveOnly)

                ALPAKA_TEST_MATH_OP_FUNCTOR(
                    OpRound,
                    Arity::Unary,
                    std::round,
                    alpaka::math::round,
                    Range::Unrestricted)

                // There is no std implementation look in Defines.hpp.
                ALPAKA_TEST_MATH_OP_FUNCTOR(
                    OpRsqrt,
                    Arity::Unary,
                    alpaka::test::unit::math::rsqrt,
                    alpaka::math::rsqrt,
                    Range::PositiveOnly)

                ALPAKA_TEST_MATH_OP_FUNCTOR(OpSin, Arity::Unary, std::sin, alpaka::math::sin, Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(
                    OpSqrt,
                    Arity::Unary,
                    std::sqrt,
                    alpaka::math::sqrt,
                    Range::PositiveAndZero)

                ALPAKA_TEST_MATH_OP_FUNCTOR(OpTan, Arity::Unary, std::tan, alpaka::math::tan, Range::Unrestricted)

                ALPAKA_TEST_MATH_OP_FUNCTOR(
                    OpTrunc,
                    Arity::Unary,
                    std::trunc,
                    alpaka::math::trunc,
                    Range::Unrestricted)

                // All binary operators.
                ALPAKA_TEST_MATH_OP_FUNCTOR(
                    OpAtan2,
                    Arity::Binary,
                    std::atan2,
                    alpaka::math::atan2,
                    Range::NotZero,
                    Range::NotZero)

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

                using BinaryFunctors = std::tuple<OpAtan2, OpFmod, OpMax, OpMin, OpPow, OpRemainder>;

                using UnaryFunctors = std::tuple<
                    OpAbs,
                    OpAcos,
                    OpAsin,
                    OpAtan,
                    OpCbrt,
                    OpCeil,
                    OpCos,
                    OpErf,
                    OpExp,
                    OpFloor,
                    OpLog,
                    OpRound,
                    OpRsqrt,
                    OpSin,
                    OpSqrt,
                    OpTan,
                    OpTrunc>;

            } // namespace math
        } // namespace unit
    } // namespace test
} // namespace alpaka
