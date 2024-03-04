/* Copyright 2023 Jiří Vyskočil, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

#include <cstdint>

namespace alpaka::rand
{
    /// Get high 32 bits of a 64-bit number
    ALPAKA_FN_HOST_ACC inline constexpr auto high32Bits(std::uint64_t const x) -> std::uint32_t
    {
        return static_cast<std::uint32_t>(x >> 32);
    }

    /// Get low 32 bits of a 64-bit number
    ALPAKA_FN_HOST_ACC inline constexpr auto low32Bits(std::uint64_t const x) -> std::uint32_t
    {
        return static_cast<std::uint32_t>(x & 0xffff'ffff);
    }

    /** Multiply two 64-bit numbers and split the result into high and low 32 bits, also known as "mulhilo32"
     *
     * @param a first 64-bit multiplier
     * @param b second 64-bit multiplier
     * @param resultHigh high 32 bits of the product a*b
     * @param resultLow low 32 bits of the product a*b
     */
    // TODO: See single-instruction implementations in original Philox source code
    ALPAKA_FN_HOST_ACC inline constexpr void multiplyAndSplit64to32(
        std::uint64_t const a,
        std::uint64_t const b,
        std::uint32_t& resultHigh,
        std::uint32_t& resultLow)
    {
        std::uint64_t res64 = a * b;
        resultHigh = high32Bits(res64);
        resultLow = low32Bits(res64);
    }
} // namespace alpaka::rand
