/* Copyright 2020 Sergei Bastrakov, Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/intrinsic/Traits.hpp>

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Fallback implementaion of popcount.
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

        //#############################################################################
        //! Fallback implementaion of ffs.
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

    //#############################################################################
    //! The Fallback intrinsic.
    class IntrinsicFallback : public concepts::Implements<ConceptIntrinsic, IntrinsicFallback>
    {
    public:
        //-----------------------------------------------------------------------------
        IntrinsicFallback() = default;
        //-----------------------------------------------------------------------------
        IntrinsicFallback(IntrinsicFallback const&) = delete;
        //-----------------------------------------------------------------------------
        IntrinsicFallback(IntrinsicFallback&&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(IntrinsicFallback const&) -> IntrinsicFallback& = delete;
        //-----------------------------------------------------------------------------
        auto operator=(IntrinsicFallback&&) -> IntrinsicFallback& = delete;
        //-----------------------------------------------------------------------------
        ~IntrinsicFallback() = default;
    };

    namespace traits
    {
        //#############################################################################
        template<>
        struct Popcount<IntrinsicFallback>
        {
            //-----------------------------------------------------------------------------
            static auto popcount(IntrinsicFallback const& /*intrinsic*/, std::uint32_t value) -> std::int32_t
            {
                return alpaka::detail::popcountFallback(value);
            }

            //-----------------------------------------------------------------------------
            static auto popcount(IntrinsicFallback const& /*intrinsic*/, std::uint64_t value) -> std::int32_t
            {
                return alpaka::detail::popcountFallback(value);
            }
        };

        //#############################################################################
        template<>
        struct Ffs<IntrinsicFallback>
        {
            //-----------------------------------------------------------------------------
            static auto ffs(IntrinsicFallback const& /*intrinsic*/, std::int32_t value) -> std::int32_t
            {
                return alpaka::detail::ffsFallback(value);
            }

            //-----------------------------------------------------------------------------
            static auto ffs(IntrinsicFallback const& /*intrinsic*/, std::int64_t value) -> std::int32_t
            {
                return alpaka::detail::ffsFallback(value);
            }
        };
    } // namespace traits
} // namespace alpaka
