/* Copyright 2024 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/platform/PlatformGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)

#    include <sycl/sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        template<>
        struct SYCLDeviceSelector<TagCpuSycl>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                return dev.is_cpu() ? 1 : -1;
            }
        };
    } // namespace detail

    //! The SYCL device manager.
    using PlatformCpuSycl = PlatformGenericSycl<TagCpuSycl>;
} // namespace alpaka

#endif
