/* Copyright 2026-2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/math/math.hpp"
#include "pmacc/math/vector/Vector.hpp"

#include <alpaka/alpaka.hpp>

#include <utility>

namespace pmacc::math
{
    /** Apply a callable element-wise to a vector.
     *
     * @param vec input vector
     * @param fn callable applied to each element
     * @return new vector with fn(vec[i]) at each index i
     */
    template<typename T_Type, uint32_t T_dim, typename T_Storage, typename T_Fn>
    constexpr auto transform(Vector<T_Type, T_dim, T_Storage> const& vec, T_Fn&& fn)
    {
        using ResultType = decltype(fn(std::declval<T_Type>()));
        Vector<ResultType, T_dim> result{};
        for(uint32_t i = 0u; i < T_dim; ++i)
            result[i] = fn(vec[i]);
        return result;
    }

    template<
        typename T_Type0,
        typename T_Type1,
        uint32_t T_dim,
        typename T_Storage0,
        typename T_Storage1,
        typename T_Fn>
    constexpr auto transform(
        Vector<T_Type0, T_dim, T_Storage0> const& a,
        Vector<T_Type1, T_dim, T_Storage1> const& b,
        T_Fn&& fn)
    {
        using ResultType = decltype(fn(std::declval<T_Type0>(), std::declval<T_Type1>()));
        Vector<ResultType, T_dim> result{};
        for(uint32_t i = 0u; i < T_dim; ++i)
            result[i] = fn(a[i], b[i]);
        return result;
    }

/** Generate an element-wise vector overload for a unary pmacc::math function.
 *
 * The scalar version of `functionName` must already be declared in pmacc::math
 * (e.g., via ALPAKA_UNARY_MATH_FN) before this macro is expanded.
 */
#define PMACC_VECTOR_UNARY_MATH_FN(functionName)                                                                      \
    template<typename T_Type, uint32_t T_dim, typename T_Storage>                                                     \
    constexpr auto functionName(Vector<T_Type, T_dim, T_Storage> const& vec)                                          \
    {                                                                                                                 \
        return transform(vec, [](auto x) { return functionName(x); });                                                \
    }

    // Log
    PMACC_VECTOR_UNARY_MATH_FN(log)
    PMACC_VECTOR_UNARY_MATH_FN(log2)
    PMACC_VECTOR_UNARY_MATH_FN(log10)

    // Exp
    PMACC_VECTOR_UNARY_MATH_FN(exp)

    // Root
    PMACC_VECTOR_UNARY_MATH_FN(sqrt)
    PMACC_VECTOR_UNARY_MATH_FN(rsqrt)
    PMACC_VECTOR_UNARY_MATH_FN(cbrt)

    // Abs
    PMACC_VECTOR_UNARY_MATH_FN(abs)

    // Trigonometric
    PMACC_VECTOR_UNARY_MATH_FN(sin)
    PMACC_VECTOR_UNARY_MATH_FN(cos)
    PMACC_VECTOR_UNARY_MATH_FN(tan)
    PMACC_VECTOR_UNARY_MATH_FN(asin)
    PMACC_VECTOR_UNARY_MATH_FN(acos)
    PMACC_VECTOR_UNARY_MATH_FN(atan)
    PMACC_VECTOR_UNARY_MATH_FN(sinh)
    PMACC_VECTOR_UNARY_MATH_FN(cosh)
    PMACC_VECTOR_UNARY_MATH_FN(tanh)
    PMACC_VECTOR_UNARY_MATH_FN(asinh)
    PMACC_VECTOR_UNARY_MATH_FN(acosh)
    PMACC_VECTOR_UNARY_MATH_FN(atanh)

    // Rounding
    PMACC_VECTOR_UNARY_MATH_FN(ceil)
    PMACC_VECTOR_UNARY_MATH_FN(floor)
    PMACC_VECTOR_UNARY_MATH_FN(trunc)
    PMACC_VECTOR_UNARY_MATH_FN(round)
    PMACC_VECTOR_UNARY_MATH_FN(lround)
    PMACC_VECTOR_UNARY_MATH_FN(llround)

    // Error functions
    PMACC_VECTOR_UNARY_MATH_FN(erf)

#undef PMACC_VECTOR_UNARY_MATH_FN

/** Generate an element-wise vector overload for a binary pmacc::math function (vec, vec).
 *
 * Both arguments must be vectors with the same element type T_Type and dimension T_dim.
 * Storage types may differ. The scalar version of `functionName` must already be declared
 * in pmacc::math before this macro is expanded.
 */
#define PMACC_VECTOR_BINARY_MATH_FN(functionName)                                                                     \
    template<typename T_Type, uint32_t T_dim, typename T_Storage0, typename T_Storage1>                               \
    constexpr auto functionName(                                                                                      \
        Vector<T_Type, T_dim, T_Storage0> const& a,                                                                   \
        Vector<T_Type, T_dim, T_Storage1> const& b)                                                                   \
    {                                                                                                                 \
        return transform(a, b, [](auto x, auto y) { return functionName(x, y); });                                    \
    }

    // Trigonometric
    PMACC_VECTOR_BINARY_MATH_FN(atan2)

    // Comparison
    PMACC_VECTOR_BINARY_MATH_FN(min)
    PMACC_VECTOR_BINARY_MATH_FN(max)

    // Modulo
    PMACC_VECTOR_BINARY_MATH_FN(fmod)
    PMACC_VECTOR_BINARY_MATH_FN(remainder)

#undef PMACC_VECTOR_BINARY_MATH_FN

/** Generate an element-wise vector overload for a binary pmacc::math function (vec, scalar).
 *
 * The first argument is a vector; the second is a scalar broadcast to all components.
 * The scalar type T_Scalar may differ from the vector element type T_Type.
 * The scalar version of `functionName` must already be declared in pmacc::math before
 * this macro is expanded.
 */
#define PMACC_VECTOR_BINARY_SCALAR_MATH_FN(functionName)                                                              \
    template<typename T_Type, uint32_t T_dim, typename T_Storage, typename T_Scalar>                                  \
    constexpr auto functionName(Vector<T_Type, T_dim, T_Storage> const& vec, T_Scalar const scalar)                   \
    {                                                                                                                 \
        return transform(vec, [scalar](auto x) { return functionName(x, scalar); });                                  \
    }

    // Pow
    PMACC_VECTOR_BINARY_SCALAR_MATH_FN(pow)

#undef PMACC_VECTOR_BINARY_SCALAR_MATH_FN

} // namespace pmacc::math
