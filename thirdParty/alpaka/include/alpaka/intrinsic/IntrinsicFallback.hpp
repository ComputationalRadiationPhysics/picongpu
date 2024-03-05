/* Copyright 2022 Sergei Bastrakov, Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/intrinsic/Traits.hpp"

namespace alpaka
{
    namespace detail
    {
        //! Fallback implementation of popcount.
        template<typename TValue>
        static auto popcountFallback(TValue value) -> std::int32_t
        {
            TValue count = 0;
            while(value != 0)
            {
                count += value & 1u;
                value >>= 1u;
            }
            return static_cast<std::int32_t>(count);
        }

        //! Fallback implementation of ffs.
        template<typename TValue>
        static auto ffsFallback(TValue value) -> std::int32_t
        {
            if(value == 0)
                return 0;
            std::int32_t result = 1;
            while((value & 1) == 0)
            {
                value >>= 1;
                result++;
            }
            return result;
        }
    } // namespace detail

    //! The Fallback intrinsic.
    class IntrinsicFallback : public concepts::Implements<ConceptIntrinsic, IntrinsicFallback>
    {
    };

    namespace trait
    {
        template<>
        struct Popcount<IntrinsicFallback>
        {
            static auto popcount(IntrinsicFallback const& /*intrinsic*/, std::uint32_t value) -> std::int32_t
            {
                return alpaka::detail::popcountFallback(value);
            }

            static auto popcount(IntrinsicFallback const& /*intrinsic*/, std::uint64_t value) -> std::int32_t
            {
                return alpaka::detail::popcountFallback(value);
            }
        };

        template<>
        struct Ffs<IntrinsicFallback>
        {
            static auto ffs(IntrinsicFallback const& /*intrinsic*/, std::int32_t value) -> std::int32_t
            {
                return alpaka::detail::ffsFallback(value);
            }

            static auto ffs(IntrinsicFallback const& /*intrinsic*/, std::int64_t value) -> std::int32_t
            {
                return alpaka::detail::ffsFallback(value);
            }
        };
    } // namespace trait
} // namespace alpaka
