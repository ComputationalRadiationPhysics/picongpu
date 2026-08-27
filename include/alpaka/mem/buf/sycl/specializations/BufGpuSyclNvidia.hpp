/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/mem/buf/sycl/BufGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU_NVIDIA)

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using ConstBufGpuSyclNvidia = ConstBufGenericSycl<TElem, TDim, TIdx, TagGpuSyclNvidia>;

    template<typename TElem, typename TDim, typename TIdx>
    using BufGpuSyclNvidia = BufGenericSycl<TElem, TDim, TIdx, TagGpuSyclNvidia>;
} // namespace alpaka

#endif
