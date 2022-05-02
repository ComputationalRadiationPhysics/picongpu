/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/pltf/PltfGenericSycl.hpp>

#    include <CL/sycl.hpp>
#    include <sycl/ext/intel/fpga_extensions.hpp>

#    include <string>

namespace alpaka::experimental
{
    //! The SYCL device manager.
    class PltfFpgaSyclIntel : public PltfGenericSycl
    {
    public:
        PltfFpgaSyclIntel() = delete;

#    ifdef ALPAKA_FPGA_EMULATION
        using selector = sycl::ext::intel::fpga_emulator_selector;
#    else
        using selector = sycl::ext::intel::fpga_selector;
#    endif
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    //! The SYCL device manager device type trait specialization.
    template<>
    struct DevType<experimental::PltfFpgaSyclIntel>
    {
        using type = experimental::DevGenericSycl<experimental::PltfFpgaSyclIntel>; // = DevFpgaSyclIntel
    };
} // namespace alpaka::trait

#endif
