/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/platform/PlatformGenericSycl.hpp"

#include <string>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include <sycl/sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        struct IntelGpuSelector
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                auto const& vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_intel_gpu = dev.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos);

                return is_intel_gpu ? 1 : -1;
            }
        };
    } // namespace detail

    //! The SYCL device manager.
    using PlatformGpuSyclIntel = PlatformGenericSycl<detail::IntelGpuSelector>;
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device manager device type trait specialization.
    template<>
    struct DevType<PlatformGpuSyclIntel>
    {
        using type = DevGenericSycl<PlatformGpuSyclIntel>; // = DevGpuSyclIntel
    };
} // namespace alpaka::trait

#endif
