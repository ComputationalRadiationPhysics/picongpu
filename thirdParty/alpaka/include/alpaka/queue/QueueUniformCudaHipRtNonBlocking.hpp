/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/queue/cuda_hip/QueueUniformCudaHipRt.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    //! The CUDA/HIP RT non-blocking queue.
    template<typename TApi>
    using QueueUniformCudaHipRtNonBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, false>;

} // namespace alpaka

#endif
