/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/kernel/TaskKernelGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)

namespace alpaka
{
    template<typename TDim, typename TIdx>
    class AccCpuSycl;

    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    using TaskKernelCpuSycl = TaskKernelGenericSycl<AccCpuSycl<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>;
} // namespace alpaka

#endif
