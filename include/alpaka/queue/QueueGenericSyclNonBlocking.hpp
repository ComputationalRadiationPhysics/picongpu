/* Copyright 2024 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/queue/sycl/QueueGenericSyclBase.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka
{
    template<typename TTag>
    using QueueGenericSyclNonBlocking = detail::QueueGenericSyclBase<TTag, false>;
} // namespace alpaka

#endif
