/* Copyright 2023 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/platform/PlatformGpuSyclIntel.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    using DevGpuSyclIntel = DevGenericSycl<PlatformGpuSyclIntel>;
} // namespace alpaka

#endif
