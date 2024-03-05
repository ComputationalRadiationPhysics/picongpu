/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dim/Traits.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/offset/Traits.hpp"

#include <cstddef>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka::detail
{
    template<typename TExtent>
    inline auto make_sycl_range(TExtent const& ext, std::size_t multiplier = 1)
    {
        constexpr auto dim = Dim<TExtent>::value;

        if constexpr(dim == 0)
            return sycl::range<1>{multiplier};
        else
        {
            auto const width = getWidth(ext) * multiplier;
            if constexpr(dim == 1)
                return sycl::range<1>{width};
            else if constexpr(dim == 2)
                return sycl::range<2>{width, getHeight(ext)};
            else
                return sycl::range<3>{width, getHeight(ext), getDepth(ext)};
        }
    }

    template<typename TView>
    inline auto make_sycl_offset(TView const& view)
    {
        constexpr auto dim = Dim<TView>::value;

        if constexpr(dim == 0)
            return sycl::range<1>{1};
        else
        {
            if constexpr(dim == 1)
                return sycl::id<1>{getOffsetX(view)};
            else if constexpr(dim == 2)
                return sycl::id<2>{getOffsetX(view), getOffsetY(view)};
            else
                return sycl::id<3>{getOffsetX(view), getOffsetY(view), getOffsetZ(view)};
        }
    }
} // namespace alpaka::detail

#endif
