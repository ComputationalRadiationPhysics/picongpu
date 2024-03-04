/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiCudaRt.hpp"
#include "alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

namespace alpaka
{
    //! The CUDA RT non-blocking queue.
    using QueueCudaRtNonBlocking = QueueUniformCudaHipRtNonBlocking<ApiCudaRt>;
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
