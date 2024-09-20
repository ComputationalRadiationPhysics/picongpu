/* Copyright 2024 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/event/EventGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    using EventGpuSyclIntel = EventGenericSycl<TagGpuSyclIntel>;
} // namespace alpaka

#endif
