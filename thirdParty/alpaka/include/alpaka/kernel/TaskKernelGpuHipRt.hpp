/* Copyright 2026 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiHipRt.hpp"
#include "alpaka/kernel/TaskKernelGpuUniformCudaHipRt.hpp"

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka
{
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, bool TCooperative, typename... TArgs>
    using TaskKernelGpuCudaHipRt
        = TaskKernelGpuUniformCudaHipRt<ApiHipRt, TAcc, TDim, TIdx, TKernelFnObj, TCooperative, TArgs...>;

} // namespace alpaka

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
