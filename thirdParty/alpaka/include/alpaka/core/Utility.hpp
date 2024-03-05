/* Copyright 2022 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/core/Common.hpp"

#include <type_traits>
#include <utility>

namespace alpaka::core
{
    //! convert any type to a reference type
    //
    // This function is equivalent to std::declval() but can be used
    // within an alpaka accelerator kernel too.
    // This function can be used only within std::decltype().
#if BOOST_LANG_CUDA && BOOST_COMP_CLANG_CUDA || BOOST_COMP_HIP
    template<class T>
    ALPAKA_FN_HOST_ACC std::add_rvalue_reference_t<T> declval();
#else
    using std::declval;
#endif

    /// Returns the ceiling of a / b, as integer.
    template<typename Integral>
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto divCeil(Integral a, Integral b) -> Integral
    {
        return (a + b - 1) / b;
    }

    /// Computes the nth power of base, in integers.
    template<typename Integral>
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto intPow(Integral base, Integral n) -> Integral
    {
        if(n == 0)
            return 1;
        auto r = base;
        for(Integral i = 1; i < n; i++)
            r *= base;
        return r;
    }

    /// Computes the floor of the nth root of value, in integers.
    template<typename Integral>
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto nthRootFloor(Integral value, Integral n) -> Integral
    {
        // adapted from: https://en.wikipedia.org/wiki/Integer_square_root
        Integral L = 0;
        Integral R = value + 1;
        while(L != R - 1)
        {
            Integral const M = (L + R) / 2;
            if(intPow(M, n) <= value)
                L = M;
            else
                R = M;
        }
        return L;
    }
} // namespace alpaka::core
