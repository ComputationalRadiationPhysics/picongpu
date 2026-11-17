/* Copyright 2026 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/acc/Tag.hpp>
#include <alpaka/warp/WarpGenericSycl.hpp>

namespace alpaka::test
{
    //! Return why the warp operations are not supported by the accelerator TAcc, or nullptr if they are supported.
    //! The warp operations are always supported, except by some SYCL backends and oneAPI versions: see
    //! alpaka/warp/WarpGenericSycl.hpp .
    template<typename TAcc>
    constexpr auto warpUnsupportedReason() -> char const*
    {
#ifdef ALPAKA_ACC_SYCL_ENABLED
        if constexpr(alpaka::accMatchesTags<
                         TAcc,
                         alpaka::TagCpuSycl,
                         alpaka::TagGpuSyclIntel,
                         alpaka::TagGpuSyclNvidia,
                         alpaka::TagGpuSyclAmd,
                         alpaka::TagFpgaSyclIntel,
                         alpaka::TagGenericSycl>)
        {
            return alpaka::warp::detail::syclWarpUnsupportedReason<alpaka::AccToTag<TAcc>>();
        }
#endif
        return nullptr;
    }
} // namespace alpaka::test
