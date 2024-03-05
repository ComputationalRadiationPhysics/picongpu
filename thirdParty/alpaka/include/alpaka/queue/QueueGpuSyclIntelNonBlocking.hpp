/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGpuSyclIntel.hpp"
#include "alpaka/queue/QueueGenericSyclNonBlocking.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    using QueueGpuSyclIntelNonBlocking = QueueGenericSyclNonBlocking<DevGpuSyclIntel>;
} // namespace alpaka

#endif
