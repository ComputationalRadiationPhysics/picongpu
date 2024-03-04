/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiCudaRt.hpp"
#include "alpaka/platform/PlatformUniformCudaHipRt.hpp"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

namespace alpaka
{
    //! The CUDA RT platform.
    using PlatformCudaRt = PlatformUniformCudaHipRt<ApiCudaRt>;
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
