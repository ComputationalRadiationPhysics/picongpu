/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/warp/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <cstdint>

namespace alpaka::experimental::warp
{
    //! The SYCL warp.
    template<typename TDim>
    class WarpGenericSycl : public concepts::Implements<alpaka::warp::ConceptWarp, WarpGenericSycl<TDim>>
    {
    public:
        WarpGenericSycl(sycl::nd_item<TDim::value> my_item) : m_item{my_item}
        {
        }

        sycl::nd_item<TDim::value> m_item;
    };
} // namespace alpaka::experimental::warp

namespace alpaka::warp::trait
{
    template<typename TDim>
    struct GetSize<experimental::warp::WarpGenericSycl<TDim>>
    {
        static auto getSize(experimental::warp::WarpGenericSycl<TDim> const& warp) -> std::int32_t
        {
            auto const sub_group = warp.m_item.get_sub_group();
            // SYCL sub-groups are always 1D
            return static_cast<std::int32_t>(sub_group.get_local_linear_range());
        }
    };

    template<typename TDim>
    struct Activemask<experimental::warp::WarpGenericSycl<TDim>>
    {
        static auto activemask(experimental::warp::WarpGenericSycl<TDim> const& warp) -> std::uint32_t
        {
            // SYCL has no way of querying this. Since sub-group functions have to be executed in convergent code
            // regions anyway we return the full mask.
            auto const sub_group = warp.m_item.get_sub_group();
            return sycl::ext::oneapi::group_ballot(sub_group, true);
        }
    };

    template<typename TDim>
    struct All<experimental::warp::WarpGenericSycl<TDim>>
    {
        static auto all(experimental::warp::WarpGenericSycl<TDim> const& warp, std::int32_t predicate) -> std::int32_t
        {
            auto const sub_group = warp.m_item.get_sub_group();
            return static_cast<std::int32_t>(sycl::all_of_group(sub_group, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct Any<experimental::warp::WarpGenericSycl<TDim>>
    {
        static auto any(experimental::warp::WarpGenericSycl<TDim> const& warp, std::int32_t predicate) -> std::int32_t
        {
            auto const sub_group = warp.m_item.get_sub_group();
            return static_cast<std::int32_t>(sycl::any_of_group(sub_group, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct Ballot<experimental::warp::WarpGenericSycl<TDim>>
    {
        static auto ballot(experimental::warp::WarpGenericSycl<TDim> const& warp, std::int32_t predicate)
        {
            auto const sub_group = warp.m_item.get_sub_group();
            return sycl::ext::oneapi::group_ballot(sub_group, static_cast<bool>(predicate));
        }
    };

    template<typename TDim>
    struct Shfl<experimental::warp::WarpGenericSycl<TDim>>
    {
        template<typename T>
        static auto shfl(
            experimental::warp::WarpGenericSycl<TDim> const& warp,
            T value,
            std::int32_t srcLane,
            std::int32_t width)
        {
            /* If width < srcLane the sub-group needs to be split into assumed subdivisions. The first item of each
               subdivision has the assumed index 0. The srcLane index is relative to the subdivisions.

               Example: If we assume a sub-group size of 32 and a width of 16 we will receive two subdivisions:
               The first starts at sub-group index 0 and the second at sub-group index 16. For srcLane = 4 the
               first subdivision will access the value at sub-group index 4 and the second at sub-group index 20. */
            auto const actual_group = warp.m_item.get_sub_group();
            auto const actual_item_id = actual_group.get_local_linear_id();

            auto const assumed_group_id = actual_item_id / width;
            auto const assumed_item_id = actual_item_id % width;

            auto const assumed_src_id = static_cast<std::size_t>(srcLane % width);
            auto const actual_src_id = assumed_src_id + assumed_group_id * width;

            auto const src = sycl::id<1>{actual_src_id};

            return sycl::select_from_group(actual_group, value, src);
        }
    };
} // namespace alpaka::warp::trait

#endif
