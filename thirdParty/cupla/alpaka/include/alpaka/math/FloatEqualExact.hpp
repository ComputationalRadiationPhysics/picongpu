/* Copyright 2021 Jiri Vyskocil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        /** Compare two floating point numbers for exact equivalence. Use only when necessary, and be aware of the
         * implications. Most codes should not use this function and instead implement a correct epsilon-based
         * comparison. If you are unfamiliar with the topic, check out
         * https://www.geeksforgeeks.org/problem-in-comparing-floating-point-numbers-and-how-to-compare-them-correctly/
         * or Goldberg 1991: "What every computer scientist should know about floating-point arithmetic",
         * https://dl.acm.org/doi/10.1145/103162.103163
         *
         * This function calls the == operator for floating point types, but disables the warning issued by the
         * compiler when compiling with the float equality warning checks enabled. This warning is valid an valuable in
         * most codes and should be generally enabled, but there are specific instances where a piece of code might
         * need to do an exact comparison (e.g. @a CudaVectorArrayWrapperTest.cpp). The verbose name for the function
         * is intentional as it should raise a red flag if used while not absolutely needed. Users are advised to add a
         * justification whenever they use this function.
         *
         * @tparam T both operands have to be the same type and conform to std::is_floating_point
         * @param a first operand
         * @param b second operand
         * @return a == b
         */
        template<typename T>
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto floatEqualExactNoWarning(T a, T b) -> bool
        {
            static_assert(std::is_floating_point_v<T>, "floatEqualExactNoWarning is for floating point values only!");

            // So far only GCC and Clang check for float comparison and both accept the GCC pragmas.
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
            return a == b;
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
        }
    } // namespace math
} // namespace alpaka
