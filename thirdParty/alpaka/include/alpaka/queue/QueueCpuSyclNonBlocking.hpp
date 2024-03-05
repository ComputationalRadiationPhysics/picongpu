/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpuSycl.hpp"
#include "alpaka/queue/QueueGenericSyclNonBlocking.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)

namespace alpaka
{
    using QueueCpuSyclNonBlocking = QueueGenericSyclNonBlocking<DevCpuSycl>;
} // namespace alpaka

#endif
