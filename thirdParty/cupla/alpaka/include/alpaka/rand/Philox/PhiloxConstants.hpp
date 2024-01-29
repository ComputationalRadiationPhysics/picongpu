/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/Philox/MultiplyAndSplit64to32.hpp"

#include <cstdint>
#include <utility>

namespace alpaka::rand::engine
{
    /** Constants used in the Philox algorithm
     *
     * The numbers are taken from the reference Philox implementation:
     *
     * J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: As easy as 1, 2, 3,"
     * SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking,
     * Storage and Analysis, 2011, pp. 1-12, doi: 10.1145/2063384.2063405.
     *
     * @tparam TParams basic Philox algorithm parameters
     *
     * static const data members are transformed into functions, because GCC
     * assumes types with static data members to be not mappable and makes not
     * exception for constexpr ones. This is a valid interpretation of the
     * OpenMP <= 4.5 standard. In OpenMP >= 5.0 types with any kind of static
     * data member are mappable.
     */
    template<typename TParams>
    class PhiloxConstants
    {
    public:
        static constexpr std::uint64_t WEYL_64_0()
        {
            return 0x9E37'79B9'7F4A'7C15; ///< First Weyl sequence parameter: the golden ratio
        }

        static constexpr std::uint64_t WEYL_64_1()
        {
            return 0xBB67'AE85'84CA'A73B; ///< Second Weyl sequence parameter: \f$ \sqrt{3}-1 \f$
        }

        static constexpr std::uint32_t WEYL_32_0()
        {
            return high32Bits(WEYL_64_0()); ///< 1st Weyl sequence parameter, 32 bits
        }

        static constexpr std::uint32_t WEYL_32_1()
        {
            return high32Bits(WEYL_64_1()); ///< 2nd Weyl sequence parameter, 32 bits
        }

        static constexpr std::uint32_t MULTIPLITER_4x32_0()
        {
            return 0xCD9E'8D57; ///< First Philox S-box multiplier
        }

        static constexpr std::uint32_t MULTIPLITER_4x32_1()
        {
            return 0xD251'1F53; ///< Second Philox S-box multiplier
        }
    };
} // namespace alpaka::rand::engine
