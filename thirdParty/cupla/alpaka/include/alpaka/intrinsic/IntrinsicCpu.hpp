/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/intrinsic/IntrinsicFallback.hpp>
#include <alpaka/intrinsic/Traits.hpp>

#include <bitset>

#if BOOST_COMP_MSVC
#    include <intrin.h>
#endif

namespace alpaka
{
    //#############################################################################
    //! The CPU intrinsic.
    class IntrinsicCpu : public concepts::Implements<ConceptIntrinsic, IntrinsicCpu>
    {
    public:
        //-----------------------------------------------------------------------------
        IntrinsicCpu() = default;
        //-----------------------------------------------------------------------------
        IntrinsicCpu(IntrinsicCpu const&) = delete;
        //-----------------------------------------------------------------------------
        IntrinsicCpu(IntrinsicCpu&&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(IntrinsicCpu const&) -> IntrinsicCpu& = delete;
        //-----------------------------------------------------------------------------
        auto operator=(IntrinsicCpu&&) -> IntrinsicCpu& = delete;
        //-----------------------------------------------------------------------------
        ~IntrinsicCpu() = default;
    };

    namespace traits
    {
        //#############################################################################
        template<>
        struct Popcount<IntrinsicCpu>
        {
            //-----------------------------------------------------------------------------
            static auto popcount(IntrinsicCpu const& /*intrinsic*/, std::uint32_t value) -> std::int32_t
            {
#if BOOST_COMP_GNUC || BOOST_COMP_CLANG || BOOST_COMP_INTEL
                return __builtin_popcount(value);
#elif BOOST_COMP_MSVC
                return __popcnt(value);
#else
                // Fallback to standard library
                return static_cast<std::int32_t>(std::bitset<32>(value).count());
#endif
            }

            //-----------------------------------------------------------------------------
            static auto popcount(IntrinsicCpu const& /*intrinsic*/, std::uint64_t value) -> std::int32_t
            {
#if BOOST_COMP_GNUC || BOOST_COMP_CLANG || BOOST_COMP_INTEL
                return __builtin_popcountll(value);
#elif BOOST_COMP_MSVC
                return static_cast<std::int32_t>(__popcnt64(value));
#else
                // Fallback to standard library
                return static_cast<std::int32_t>(std::bitset<64>(value).count());
#endif
            }
        };

        //#############################################################################
        template<>
        struct Ffs<IntrinsicCpu>
        {
            //-----------------------------------------------------------------------------
            static auto ffs(IntrinsicCpu const& /*intrinsic*/, std::int32_t value) -> std::int32_t
            {
#if BOOST_COMP_GNUC || BOOST_COMP_CLANG || BOOST_COMP_INTEL
                return __builtin_ffs(value);
#elif BOOST_COMP_MSVC
                // Implementation based on
                // https://gitlab.freedesktop.org/cairo/cairo/commit/f5167dc2e1a13d8c4e5d66d7178a24b9b5e7ac7a
                unsigned long index = 0u;
                if(_BitScanForward(&index, value) != 0)
                    return static_cast<std::int32_t>(index + 1u);
                else
                    return 0;
#else
                return alpaka::detail::ffsFallback(value);
#endif
            }

            //-----------------------------------------------------------------------------
            static auto ffs(IntrinsicCpu const& /*intrinsic*/, std::int64_t value) -> std::int32_t
            {
#if BOOST_COMP_GNUC || BOOST_COMP_CLANG || BOOST_COMP_INTEL
                return __builtin_ffsll(value);
#elif BOOST_COMP_MSVC
                // Implementation based on
                // https://gitlab.freedesktop.org/cairo/cairo/commit/f5167dc2e1a13d8c4e5d66d7178a24b9b5e7ac7a
                unsigned long index = 0u;
                if(_BitScanForward64(&index, value) != 0)
                    return static_cast<std::int32_t>(index + 1u);
                else
                    return 0;
#else
                return alpaka::detail::ffsFallback(value);
#endif
            }
        };
    } // namespace traits
} // namespace alpaka
