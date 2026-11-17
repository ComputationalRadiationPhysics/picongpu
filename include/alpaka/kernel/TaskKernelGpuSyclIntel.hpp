/* Copyright 2026 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGpuSyclIntel.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/kernel/TaskKernelGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, bool TCooperative, typename... TArgs>
    using TaskKernelGpuSyclIntel = TaskKernelGenericSycl<
        TagGpuSyclIntel,
        AccGpuSyclIntel<TDim, TIdx>,
        TDim,
        TIdx,
        TKernelFnObj,
        TCooperative,
        TArgs...>;

} // namespace alpaka

#endif
