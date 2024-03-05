/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpuSycl.hpp"
#include "alpaka/event/EventGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)

namespace alpaka
{
    using EventCpuSycl = EventGenericSycl<DevCpuSycl>;
} // namespace alpaka

#endif
