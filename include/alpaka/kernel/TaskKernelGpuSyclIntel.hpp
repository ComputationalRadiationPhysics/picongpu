/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/kernel/TaskKernelGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    template<typename TDim, typename TIdx>
    class AccGpuSyclIntel;

    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    using TaskKernelGpuSyclIntel
        = TaskKernelGenericSycl<AccGpuSyclIntel<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>;
} // namespace alpaka

#endif
