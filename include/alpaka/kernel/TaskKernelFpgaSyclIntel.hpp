/* Copyright 2026 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccFpgaSyclIntel.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/kernel/TaskKernelGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, bool TCooperative, typename... TArgs>
    using TaskKernelFpgaSyclIntel = TaskKernelGenericSycl<
        TagFpgaSyclIntel,
        AccFpgaSyclIntel<TDim, TIdx>,
        TDim,
        TIdx,
        TKernelFnObj,
        TCooperative,
        TArgs...>;

} // namespace alpaka

#endif
