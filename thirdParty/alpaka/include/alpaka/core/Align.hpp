/* Copyright 2022 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

#include <cstddef>
#include <type_traits>

namespace alpaka::core
{
    //! Rounds to the next higher power of two (if not already power of two).
    // Adapted from llvm/ADT/SmallPtrSet.h
    template<std::size_t N>
    struct RoundUpToPowerOfTwo;

    //! Defines implementation details that should not be used directly by the user.
    namespace detail
    {
        //! Base case for N being a power of two.
        template<std::size_t N, bool TisPowerTwo>
        struct RoundUpToPowerOfTwoHelper : std::integral_constant<std::size_t, N>
        {
        };

        //! Case for N not being a power of two.
        // We could just use NextVal = N+1, but this converges faster.  N|(N-1) sets
        // the right-most zero bits to one all at once, e.g. 0b0011000 -> 0b0011111.
        template<std::size_t N>
        struct RoundUpToPowerOfTwoHelper<N, false>
            : std::integral_constant<std::size_t, RoundUpToPowerOfTwo<(N | (N - 1)) + 1>::value>
        {
        };
    } // namespace detail

    template<std::size_t N>
    struct RoundUpToPowerOfTwo
        : std::integral_constant<std::size_t, detail::RoundUpToPowerOfTwoHelper<N, (N & (N - 1)) == 0>::value>
    {
    };

    //! The alignment specifics.
    namespace align
    {
        //! Calculates the optimal alignment for data of the given size.
        template<std::size_t TsizeBytes>
        struct OptimalAlignment
            : std::integral_constant<
                  std::size_t,
#if BOOST_COMP_GNUC
                  // GCC does not support alignments larger then 128: "warning: requested alignment 256 is larger
                  // than 128[-Wattributes]".
                  (TsizeBytes > 64) ? 128 :
#endif
                                    (RoundUpToPowerOfTwo<TsizeBytes>::value)>
        {
        };
    } // namespace align
} // namespace alpaka::core

// The optimal alignment for a type is the next higher or equal power of two.
#define ALPAKA_OPTIMAL_ALIGNMENT(...)                                                                                 \
    ::alpaka::core::align::OptimalAlignment<sizeof(std::remove_cv_t<__VA_ARGS__>)>::value
