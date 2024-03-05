/* Copyright 2023 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGpuSyclIntel.hpp"
#include "alpaka/event/EventGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    using EventGpuSyclIntel = EventGenericSycl<DevGpuSyclIntel>;
} // namespace alpaka

#endif
