/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/pltf/PltfGenericSycl.hpp>

#    include <CL/sycl.hpp>

#    include <string>

namespace alpaka::experimental
{
    namespace detail
    {
        // Prevent clang from annoying us with warnings about emitting too many vtables. These are discarded by
        // the linker anyway.
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wweak-vtables"
        struct IntelGpuSelector : sycl::device_selector
#    endif
        {
            auto operator()(sycl::device const& dev) const -> int override
            {
                auto const vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_intel_gpu = (vendor.find("Intel(R) Corporation") != std::string::npos) && dev.is_gpu();

                return is_intel_gpu ? 1 : -1;
            }
        };
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
    } // namespace detail

    //! The SYCL device manager.
    class PltfGpuSyclIntel : public PltfGenericSycl
    {
    public:
        PltfGpuSyclIntel() = delete;

        using selector = detail::IntelGpuSelector;
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    //! The SYCL device manager device type trait specialization.
    template<>
    struct DevType<experimental::PltfGpuSyclIntel>
    {
        using type = experimental::DevGenericSycl<experimental::PltfGpuSyclIntel>; // = DevGpuSyclIntel
    };
} // namespace alpaka::trait

#endif
