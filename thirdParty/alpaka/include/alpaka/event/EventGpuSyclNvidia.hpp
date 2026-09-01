/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/event/EventGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU_NVIDIA)

namespace alpaka
{
    using EventGpuSyclNvidia = EventGenericSycl<TagGpuSyclNvidia>;
} // namespace alpaka

#endif
