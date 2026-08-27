/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/mem/buf/sycl/BufGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU_AMD)

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufGpuSyclAmd = ConstBufGenericSycl<TElem, TDim, TIdx, TagGpuSyclAmd>;

    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclAmd = BufGenericSycl<TElem, TDim, TIdx, TagGpuSyclAmd>;
} // namespace alpaka

#endif
