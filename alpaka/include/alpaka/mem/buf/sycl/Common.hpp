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

#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/offset/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <cstddef>

namespace alpaka::experimental::detail
{
    template<typename TExtent>
    inline auto make_sycl_range(TExtent const& ext, std::size_t multiplier = 1)
    {
        constexpr auto dim = Dim<TExtent>::value;

        auto const width = getWidth(ext) * multiplier;

        if constexpr(dim == 1)
            return sycl::range<1>{width};
        else if constexpr(dim == 2)
            return sycl::range<2>{width, getHeight(ext)};
        else
            return sycl::range<3>{width, getHeight(ext), getDepth(ext)};
    }

    template<typename TView>
    inline auto make_sycl_offset(TView const& view)
    {
        constexpr auto dim = Dim<TView>::value;

        if constexpr(dim == 1)
            return sycl::id<1>{getOffsetX(view)};
        else if constexpr(dim == 2)
            return sycl::id<2>{getOffsetX(view), getOffsetY(view)};
        else
            return sycl::id<3>{getOffsetX(view), getOffsetY(view), getOffsetZ(view)};
    }
} // namespace alpaka::experimental::detail

#endif
