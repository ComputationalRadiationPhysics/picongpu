/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/warp/Traits.hpp"

#include <cstdint>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka::warp
{
    //! The SYCL warp.
    template<typename TDim>
    class WarpGenericSycl : public concepts::Implements<alpaka::warp::ConceptWarp, WarpGenericSycl<TDim>>
    {
    public:
        WarpGenericSycl(sycl::nd_item<TDim::value> my_item) : m_item_warp{my_item}
        {
        }

        sycl::nd_item<TDim::value> m_item_warp;
    };
} // namespace alpaka::warp

namespace alpaka::warp::trait
{
    template<typename TDim>
    struct GetSize<warp::WarpGenericSycl<TDim>>
    {
        static auto getSize(warp::WarpGenericSycl<TDim> const& warp) -> std::int32_t
        {
            auto const sub_group = warp.m_item_warp.get_sub_group();
            // SYCL sub-groups are always 1D
            return static_cast<std::int32_t>(sub_group.get_max_local_range()[0]);
        }
    };

    template<typename TDim>
    struct Activemask<warp::WarpGenericSycl<TDim>>
    {
        // FIXME This should be std::uint64_t on AMD GCN architectures and on CPU,
        // but the former is not targeted in alpaka and CPU case is not supported in SYCL yet.
        // Restrict to warpSize <= 32 for now.
        static auto activemask(warp::WarpGenericSycl<TDim> const& warp) -> std::uint32_t
        {
            // SYCL has no way of querying this. Since sub-group functions have to be executed in convergent code
            // regions anyway we return the full mask.
            auto const sub_group = warp.m_item_warp.get_sub_group();
            auto const mask = sycl::ext::oneapi::group_ballot(sub_group, true);
            // FIXME This should be std::uint64_t on AMD GCN architectures and on CPU,
            // but the former is not targeted in alpaka and CPU case is not supported in SYCL yet.
            // Restrict to warpSize <= 32 for now.
            std::uint32_t bits = 0;
            mask.extract_bits(bits);
            return bits;
        }
    };

    template<typename TDim>
    struct All<warp::WarpGenericSycl<TDim>>
    {
        static auto all(warp::WarpGenericSycl<TDim> const& warp, std::int32_t predicate) -> std::int32_t
        {
            auto const sub_group = warp.m_item_warp.get_sub_group();
            return static_cast<std::int32_t>(sycl::all_of_group(sub_group, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct Any<warp::WarpGenericSycl<TDim>>
    {
        static auto any(warp::WarpGenericSycl<TDim> const& warp, std::int32_t predicate) -> std::int32_t
        {
            auto const sub_group = warp.m_item_warp.get_sub_group();
            return static_cast<std::int32_t>(sycl::any_of_group(sub_group, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct Ballot<warp::WarpGenericSycl<TDim>>
    {
        // FIXME This should be std::uint64_t on AMD GCN architectures and on CPU,
        // but the former is not targeted in alpaka and CPU case is not supported in SYCL yet.
        // Restrict to warpSize <= 32 for now.
        static auto ballot(warp::WarpGenericSycl<TDim> const& warp, std::int32_t predicate) -> std::uint32_t
        {
            auto const sub_group = warp.m_item_warp.get_sub_group();
            auto const mask = sycl::ext::oneapi::group_ballot(sub_group, static_cast<bool>(predicate));
            // FIXME This should be std::uint64_t on AMD GCN architectures and on CPU,
            // but the former is not targeted in alpaka and CPU case is not supported in SYCL yet.
            // Restrict to warpSize <= 32 for now.
            std::uint32_t bits = 0;
            mask.extract_bits(bits);
            return bits;
        }
    };

    template<typename TDim>
    struct Shfl<warp::WarpGenericSycl<TDim>>
    {
        template<typename T>
        static auto shfl(warp::WarpGenericSycl<TDim> const& warp, T value, std::int32_t srcLane, std::int32_t width)
        {
            ALPAKA_ASSERT_OFFLOAD(width > 0);
            ALPAKA_ASSERT_OFFLOAD(srcLane < width);
            ALPAKA_ASSERT_OFFLOAD(srcLane >= 0);

            /* If width < srcLane the sub-group needs to be split into assumed subdivisions. The first item of each
               subdivision has the assumed index 0. The srcLane index is relative to the subdivisions.

               Example: If we assume a sub-group size of 32 and a width of 16 we will receive two subdivisions:
               The first starts at sub-group index 0 and the second at sub-group index 16. For srcLane = 4 the
               first subdivision will access the value at sub-group index 4 and the second at sub-group index 20. */
            auto const actual_group = warp.m_item_warp.get_sub_group();
            auto const actual_item_id = static_cast<std::int32_t>(actual_group.get_local_linear_id());
            auto const actual_group_id = actual_item_id / width;
            auto const actual_src_id = static_cast<std::size_t>(srcLane + actual_group_id * width);
            auto const src = sycl::id<1>{actual_src_id};

            return sycl::select_from_group(actual_group, value, src);
        }
    };
} // namespace alpaka::warp::trait

#endif
