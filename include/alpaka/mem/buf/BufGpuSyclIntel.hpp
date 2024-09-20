/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGpuSyclIntel.hpp"
#include "alpaka/mem/buf/BufGenericSycl.hpp"
#include "alpaka/platform/PlatformGpuSyclIntel.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclIntel = BufGenericSycl<TElem, TDim, TIdx, PlatformGpuSyclIntel>;
} // namespace alpaka

#endif
