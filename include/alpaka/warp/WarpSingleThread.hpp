/* Copyright 2022 Sergei Bastrakov, David M. Rogers, Bernhard Manfred Gruber, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/warp/Traits.hpp"

#include <cstdint>

namespace alpaka::warp
{
    //! The single-threaded warp to emulate it on CPUs.
    class WarpSingleThread : public concepts::Implements<ConceptWarp, WarpSingleThread>
    {
    };

    namespace trait
    {
        template<>
        struct GetSize<WarpSingleThread>
        {
            static auto getSize(warp::WarpSingleThread const& /*warp*/)
            {
                return 1;
            }
        };

        template<>
        struct Activemask<WarpSingleThread>
        {
            static auto activemask(warp::WarpSingleThread const& /*warp*/)
            {
                return 1u;
            }
        };

        template<>
        struct All<WarpSingleThread>
        {
            static auto all(warp::WarpSingleThread const& /*warp*/, std::int32_t predicate)
            {
                return predicate;
            }
        };

        template<>
        struct Any<WarpSingleThread>
        {
            static auto any(warp::WarpSingleThread const& /*warp*/, std::int32_t predicate)
            {
                return predicate;
            }
        };

        template<>
        struct Ballot<WarpSingleThread>
        {
            static auto ballot(warp::WarpSingleThread const& /*warp*/, std::int32_t predicate)
            {
                return predicate ? 1u : 0u;
            }
        };

        template<>
        struct Shfl<WarpSingleThread>
        {
            template<typename T>
            static auto shfl(
                warp::WarpSingleThread const& /*warp*/,
                T val,
                std::int32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }
        };

        template<>
        struct ShflUp<WarpSingleThread>
        {
            template<typename T>
            static auto shfl_up(
                warp::WarpSingleThread const& /*warp*/,
                T val,
                std::uint32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }
        };

        template<>
        struct ShflDown<WarpSingleThread>
        {
            template<typename T>
            static auto shfl_down(
                warp::WarpSingleThread const& /*warp*/,
                T val,
                std::uint32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }
        };

        template<>
        struct ShflXor<WarpSingleThread>
        {
            template<typename T>
            static auto shfl_xor(
                warp::WarpSingleThread const& /*warp*/,
                T val,
                std::int32_t /*srcLane*/,
                std::int32_t /*width*/)
            {
                return val;
            }
        };
    } // namespace trait
} // namespace alpaka::warp
