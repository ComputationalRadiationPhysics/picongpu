/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/warp/Traits.hpp>

#include <cstdint>

namespace alpaka
{
    namespace warp
    {
        //#############################################################################
        //! The single-threaded warp to emulate it on CPUs.
        class WarpSingleThread : public concepts::Implements<ConceptWarp, WarpSingleThread>
        {
        public:
            //-----------------------------------------------------------------------------
            WarpSingleThread() = default;
            //-----------------------------------------------------------------------------
            WarpSingleThread(WarpSingleThread const&) = delete;
            //-----------------------------------------------------------------------------
            WarpSingleThread(WarpSingleThread&&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WarpSingleThread const&) -> WarpSingleThread& = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WarpSingleThread&&) -> WarpSingleThread& = delete;
            //-----------------------------------------------------------------------------
            ~WarpSingleThread() = default;
        };

        namespace traits
        {
            //#############################################################################
            template<>
            struct GetSize<WarpSingleThread>
            {
                //-----------------------------------------------------------------------------
                static auto getSize(warp::WarpSingleThread const& /*warp*/)
                {
                    return 1;
                }
            };

            //#############################################################################
            template<>
            struct Activemask<WarpSingleThread>
            {
                //-----------------------------------------------------------------------------
                static auto activemask(warp::WarpSingleThread const& /*warp*/)
                {
                    return 1u;
                }
            };

            //#############################################################################
            template<>
            struct All<WarpSingleThread>
            {
                //-----------------------------------------------------------------------------
                static auto all(warp::WarpSingleThread const& /*warp*/, std::int32_t predicate)
                {
                    return predicate;
                }
            };

            //#############################################################################
            template<>
            struct Any<WarpSingleThread>
            {
                //-----------------------------------------------------------------------------
                static auto any(warp::WarpSingleThread const& /*warp*/, std::int32_t predicate)
                {
                    return predicate;
                }
            };

            //#############################################################################
            template<>
            struct Ballot<WarpSingleThread>
            {
                //-----------------------------------------------------------------------------
                static auto ballot(warp::WarpSingleThread const& /*warp*/, std::int32_t predicate)
                {
                    return predicate ? 1u : 0u;
                }
            };
        } // namespace traits
    } // namespace warp
} // namespace alpaka
