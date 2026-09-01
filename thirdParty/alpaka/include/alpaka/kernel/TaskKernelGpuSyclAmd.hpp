/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/kernel/TaskKernelGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU_AMD)

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    using TaskKernelGpuSyclAmd
        = TaskKernelGenericSycl<TagGpuSyclAmd, AccGpuSyclAmd<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>;

} // namespace alpaka

#endif
