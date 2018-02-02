/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <boost/predef.h>

#include <cstddef>
#include <type_traits>

namespace alpaka
{
    namespace core
    {
        //-----------------------------------------------------------------------------
        //! Rounds to the next higher power of two (if not already power of two).
        // Adapted from llvm/ADT/SmallPtrSet.h
        template<
            std::size_t N>
        struct RoundUpToPowerOfTwo;

        //-----------------------------------------------------------------------------
        //! Defines implementation details that should not be used directly by the user.
        namespace detail
        {
            //-----------------------------------------------------------------------------
            //! Base case for N being a power of two.
            template<
                std::size_t N,
                bool TisPowerTwo>
            struct RoundUpToPowerOfTwoHelper :
                std::integral_constant<
                    std::size_t,
                    N>
            {};
            //-----------------------------------------------------------------------------
            //! Case for N not being a power of two.
            // We could just use NextVal = N+1, but this converges faster.  N|(N-1) sets
            // the right-most zero bits to one all at once, e.g. 0b0011000 -> 0b0011111.
            template<
                std::size_t N>
            struct RoundUpToPowerOfTwoHelper<
                N,
                false> :
                    std::integral_constant<
                        std::size_t,
                        RoundUpToPowerOfTwo<(N | (N - 1)) + 1>::value>
            {};
        }
        //-----------------------------------------------------------------------------
        template<
            std::size_t N>
        struct RoundUpToPowerOfTwo :
            std::integral_constant<
                std::size_t,
                detail::RoundUpToPowerOfTwoHelper<
                    N,
                    (N&(N - 1)) == 0>::value>
        {};

        //-----------------------------------------------------------------------------
        //! The alignment specifics.
        namespace align
        {
            //-----------------------------------------------------------------------------
            //! Calculates the optimal alignment for data of the given size.
            template<
                std::size_t TsizeBytes>
            struct OptimalAlignment :
                std::integral_constant<
                    std::size_t,
#if BOOST_COMP_GNUC
                    // GCC does not support alignments larger then 128: "warning: requested alignment 256 is larger than 128[-Wattributes]".
                    (TsizeBytes > 64)
                        ? 128
                        :
#endif
                            (RoundUpToPowerOfTwo<TsizeBytes>::value)>
            {};
        }
    }
}

// ICC does not support constant expressions as parameters to alignas
// The optimal alignment for a type is the next higher or equal power of two.
#if BOOST_COMP_INTEL
    #define ALPAKA_OPTIMAL_ALIGNMENT_SIZE(...)\
            ((__VA_ARGS__)==1?1:\
            ((__VA_ARGS__)<=2?2:\
            ((__VA_ARGS__)<=4?4:\
            ((__VA_ARGS__)<=8?8:\
            ((__VA_ARGS__)<=16?16:\
            ((__VA_ARGS__)<=32?32:\
            ((__VA_ARGS__)<=64?64:128\
            )))))))
    #define ALPAKA_OPTIMAL_ALIGNMENT(...)\
            ALPAKA_OPTIMAL_ALIGNMENT_SIZE(sizeof(typename std::remove_cv<__VA_ARGS__>::type))
#else
    #define ALPAKA_OPTIMAL_ALIGNMENT(...)\
            ::alpaka::core::align::OptimalAlignment<sizeof(typename std::remove_cv<__VA_ARGS__>::type)>::value
#endif
