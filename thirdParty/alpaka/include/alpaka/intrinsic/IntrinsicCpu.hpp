/* Copyright 2023 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/intrinsic/IntrinsicFallback.hpp"
#include "alpaka/intrinsic/Traits.hpp"

#include <bitset>
#include <climits>
#if __has_include(<version>) // Not part of the C++17 standard but all major standard libraries include this
#    include <version>
#endif
#ifdef __cpp_lib_bitops
#    include <bit>
#endif

#if BOOST_COMP_MSVC
#    include <intrin.h>
#endif

namespace alpaka
{
    //! The CPU intrinsic.
    class IntrinsicCpu : public concepts::Implements<ConceptIntrinsic, IntrinsicCpu>
    {
    };

    namespace trait
    {
        template<>
        struct Popcount<IntrinsicCpu>
        {
            template<typename UnsignedIntegral>
            static auto popcount(IntrinsicCpu const& /*intrinsic*/, UnsignedIntegral value) -> std::int32_t
            {
#ifdef __cpp_lib_bitops
                return std::popcount(value);
#elif BOOST_COMP_GNUC || BOOST_COMP_CLANG
                if constexpr(sizeof(UnsignedIntegral) == 8)
                    return __builtin_popcountll(value);
                else
                    return __builtin_popcount(value);
#elif BOOST_COMP_MSVC
                if constexpr(sizeof(UnsignedIntegral) == 8)
                    return static_cast<std::int32_t>(__popcnt64(value));
                else
                    return __popcnt(value);
#else
                // Fallback to standard library
                return static_cast<std::int32_t>(std::bitset<sizeof(UnsignedIntegral) * CHAR_BIT>(value).count());
#endif
                ALPAKA_UNREACHABLE(0);
            }
        };

        template<>
        struct Ffs<IntrinsicCpu>
        {
            template<typename Integral>
            static auto ffs(IntrinsicCpu const& /*intrinsic*/, Integral value) -> std::int32_t
            {
#ifdef __cpp_lib_bitops
                return value == 0 ? 0 : std::countr_zero(static_cast<std::make_unsigned_t<Integral>>(value)) + 1;
#elif BOOST_COMP_GNUC || BOOST_COMP_CLANG
                if constexpr(sizeof(Integral) == 8)
                    return __builtin_ffsll(value);
                else
                    return __builtin_ffs(value);
#elif BOOST_COMP_MSVC
                // Implementation based on
                // https://gitlab.freedesktop.org/cairo/cairo/commit/f5167dc2e1a13d8c4e5d66d7178a24b9b5e7ac7a
                unsigned long index = 0u;
                if constexpr(sizeof(Integral) == 8)
                    return _BitScanForward64(&index, value) == 0 ? 0 : static_cast<std::int32_t>(index + 1u);
                else
                    return _BitScanForward(&index, value) == 0 ? 0 : static_cast<std::int32_t>(index + 1u);
#else
                return alpaka::detail::ffsFallback(value);
#endif
                ALPAKA_UNREACHABLE(0);
            }
        };
    } // namespace trait
} // namespace alpaka
