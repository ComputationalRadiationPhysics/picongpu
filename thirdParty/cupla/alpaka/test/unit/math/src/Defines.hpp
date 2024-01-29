/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Sergei Bastrakov
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace mathtest
{
    // New types need to be added to the switch-case in DataGen.hpp
    enum class Range
    {
        OneNeighbourhood,
        PositiveOnly,
        PositiveAndZero,
        NotZero,
        Unrestricted,
        Anything
    };

    // New types need to be added to the operator() function in Functor.hpp
    enum class Arity
    {
        Unary = 1,
        Binary = 2,
        Ternary = 3
    };

    template<typename T, Arity Tarity>
    struct ArgsItem
    {
        static constexpr Arity arity = Tarity;
        static constexpr size_t arity_nr = static_cast<size_t>(Tarity);

        T arg[arity_nr]; // represents arg0, arg1, ...

        friend auto operator<<(std::ostream& os, ArgsItem const& argsItem) -> std::ostream&
        {
            os.precision(17);
            for(size_t i = 0; i < argsItem.arity_nr; ++i)
                os << (i == 0 ? "[ " : ", ") << std::setprecision(std::numeric_limits<T>::digits10 + 1)
                   << argsItem.arg[i];
            os << " ]";
            return os;
        }
    };

    //! Reference implementation of rsqrt, since there is no std::rsqrt
    template<typename T>
    auto rsqrt(T const& arg)
    {
        // Need ADL for complex numbers
        using std::sqrt;
        return static_cast<T>(1) / sqrt(arg);
    }

    //! Stub for division expressed same way as alpaka math traits
    template<typename TAcc, typename T>
    ALPAKA_FN_HOST_ACC auto divides(TAcc&, T const& arg1, T const& arg2)
    {
        return arg1 / arg2;
    }

    //! Stub for subtraction expressed same way as alpaka math traits
    template<typename TAcc, typename T>
    ALPAKA_FN_HOST_ACC auto minus(TAcc&, T const& arg1, T const& arg2)
    {
        return arg1 - arg2;
    }

    //! Stub for multiplication expressed same way as alpaka math traits
    template<typename TAcc, typename T>
    ALPAKA_FN_HOST_ACC auto multiplies(TAcc&, T const& arg1, T const& arg2)
    {
        return arg1 * arg2;
    }

    //! Stub for addition expressed same way as alpaka math traits
    template<typename TAcc, typename T>
    ALPAKA_FN_HOST_ACC auto plus(TAcc&, T const& arg1, T const& arg2)
    {
        return arg1 + arg2;
    }

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
} // namespace mathtest
