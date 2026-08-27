/* Copyright 2026 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 *
 * The implementations of Shfl::shfl(), ShflUp::shfl_up(), ShflDown::shfl_down() and ShflXor::shfl_xor() are derived
 * from Intel DPCT.
 * Copyright (C) Intel Corporation.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * See https://llvm.org/LICENSE.txt for license information.
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Config.hpp"
#include "alpaka/warp/Traits.hpp"

#include <cstdint>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka::warp
{
    //! The SYCL warp.
    template<typename TDim>
    class WarpGenericSycl : public interface::Implements<alpaka::warp::ConceptWarp, WarpGenericSycl<TDim>>
    {
    public:
        using mask_type = std::uint32_t;

        WarpGenericSycl(sycl::nd_item<TDim::value> my_item) : m_item_warp{my_item}
        {
        }

        sycl::nd_item<TDim::value> m_item_warp;
    };
} // namespace alpaka::warp

namespace alpaka::warp::trait
{
    // oneAPI up to 2025.3 uses sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group(),
    // while oneAPI 2026.0 uses sycl::ext::oneapi::experimental::this_work_item::get_opportunistic_group()
#    if ALPAKA_COMP_ICPX >= ALPAKA_VERSION_NUMBER(2026, 0, 0)
    using sycl::ext::oneapi::experimental::this_work_item::get_opportunistic_group;
#    else
    using sycl::ext::oneapi::experimental::this_kernel::get_opportunistic_group;
#    endif

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
    struct GetSizeCompileTime<warp::WarpGenericSycl<TDim>>
    {
        static constexpr auto getSizeCompileTime() -> std::int32_t
        {
            // SYCL sub-groups size is usually not known at compile time
            return 0;
        }
    };

    template<typename TDim>
    struct GetSizeUpperLimit<warp::WarpGenericSycl<TDim>>
    {
        static constexpr auto getSizeUpperLimit() -> std::int32_t
        {
            // See include/alpaka/kernel/SyclSubgroupSize.hpp for possible sub-group sizes.
            return 64;
        }
    };

    template<typename TDim>
    struct Activemask<warp::WarpGenericSycl<TDim>>
    {
        // FIXME This should be std::uint64_t on AMD GCN architectures and on CPU,
        // but the former is not targeted in alpaka and CPU case is not supported in SYCL yet.
        // Restrict to warpSize <= 32 for now.
        static auto activemask(warp::WarpGenericSycl<TDim> const& /*warp*/) -> warp::WarpGenericSycl<TDim>::mask_type
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            auto const mask = sycl::ext::oneapi::group_ballot(sg, true);
            std::uint32_t bits = 0;
            mask.extract_bits(bits);
            return bits;
        }
    };

    template<typename TDim>
    struct All<warp::WarpGenericSycl<TDim>>
    {
        static auto all(warp::WarpGenericSycl<TDim> const& /*warp*/, std::int32_t predicate) -> std::int32_t
        {
            auto activegroup = get_opportunistic_group();
            return static_cast<std::int32_t>(sycl::all_of_group(activegroup, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct Any<warp::WarpGenericSycl<TDim>>
    {
        static auto any(warp::WarpGenericSycl<TDim> const& /*warp*/, std::int32_t predicate) -> std::int32_t
        {
            auto activegroup = get_opportunistic_group();
            return static_cast<std::int32_t>(sycl::any_of_group(activegroup, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct Ballot<warp::WarpGenericSycl<TDim>>
    {
        // FIXME This should be std::uint64_t on AMD GCN architectures and on CPU,
        // but the former is not targeted in alpaka and CPU case is not supported in SYCL yet.
        // Restrict to warpSize <= 32 for now.
        static auto ballot(warp::WarpGenericSycl<TDim> const& /*warp*/, std::int32_t predicate)
            -> warp::WarpGenericSycl<TDim>::mask_type
        {
            auto sub_group = sycl::ext::oneapi::this_work_item::get_sub_group();
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
        static auto shfl(
            warp::WarpGenericSycl<TDim> const& /*warp*/,
            T value,
            std::int32_t srcLane,
            std::int32_t width)
        {
            ALPAKA_ASSERT_ACC(width > 0);
            ALPAKA_ASSERT_ACC(srcLane >= 0);

            /* If width < srcLane the sub-group needs to be split into assumed subdivisions. The first item of each
               subdivision has the assumed index 0. The srcLane index is relative to the subdivisions.

               Example: If we assume a sub-group size of 32 and a width of 16 we will receive two subdivisions:
               The first starts at sub-group index 0 and the second at sub-group index 16. For srcLane = 4 the
               first subdivision will access the value at sub-group index 4 and the second at sub-group index 20. */
            auto actual_group = get_opportunistic_group();
            std::uint32_t const w = static_cast<std::uint32_t>(width);
            std::uint32_t const start_index = actual_group.get_local_linear_id() / w * w;
            return sycl::select_from_group(actual_group, value, start_index + static_cast<std::uint32_t>(srcLane) % w);
        }
    };

    template<typename TDim>
    struct ShflUp<warp::WarpGenericSycl<TDim>>
    {
        template<typename T>
        static auto shfl_up(
            warp::WarpGenericSycl<TDim> const& /*warp*/,
            T value,
            std::uint32_t offset, /* must be the same for all work-items in the group */
            std::int32_t width)
        {
            auto actual_group = get_opportunistic_group();
            std::uint32_t const w = static_cast<std::uint32_t>(width);
            std::uint32_t const id = actual_group.get_local_linear_id();
            std::uint32_t const start_index = id / w * w;
            T result = sycl::shift_group_right(actual_group, value, offset);
            if((id - start_index) < offset)
            {
                result = value;
            }
            return result;
        }
    };

    template<typename TDim>
    struct ShflDown<warp::WarpGenericSycl<TDim>>
    {
        template<typename T>
        static auto shfl_down(
            warp::WarpGenericSycl<TDim> const& /*warp*/,
            T value,
            std::uint32_t offset,
            std::int32_t width)
        {
            auto actual_group = get_opportunistic_group();
            std::uint32_t const w = static_cast<std::uint32_t>(width);
            std::uint32_t const id = actual_group.get_local_linear_id();
            std::uint32_t const end_index = (id / w + 1) * w;
            T result = sycl::shift_group_left(actual_group, value, offset);
            if((id + offset) >= end_index)
            {
                result = value;
            }
            return result;
        }
    };

    template<typename TDim>
    struct ShflXor<warp::WarpGenericSycl<TDim>>
    {
        template<typename T>
        static auto shfl_xor(
            warp::WarpGenericSycl<TDim> const& /*warp*/,
            T value,
            std::int32_t mask,
            std::int32_t width)
        {
            auto actual_group = get_opportunistic_group();
            std::uint32_t const w = static_cast<std::uint32_t>(width);
            std::uint32_t const id = actual_group.get_local_linear_id();
            std::uint32_t const start_index = id / w * w;
            std::uint32_t const target_offset = (id % w) ^ static_cast<std::uint32_t>(mask);
            return sycl::select_from_group(actual_group, value, target_offset < w ? start_index + target_offset : id);
        }
    };
} // namespace alpaka::warp::trait

#endif
