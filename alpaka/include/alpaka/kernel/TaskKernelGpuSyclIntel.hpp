/* Copyright 2024 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/kernel/TaskKernelGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    using TaskKernelGpuSyclIntel
        = TaskKernelGenericSycl<TagGpuSyclIntel, AccGpuSyclIntel<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>;

} // namespace alpaka

#endif
