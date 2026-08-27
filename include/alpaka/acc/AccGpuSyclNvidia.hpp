/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Sycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU_NVIDIA)

namespace alpaka
{
    //! The Nvidia GPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Nvidia GPU target device.
    template<typename TDim, typename TIdx>
    using AccGpuSyclNvidia = AccGenericSycl<TagGpuSyclNvidia, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccGpuSyclNvidia<TDim, TIdx>>
        {
            using type = alpaka::TagGpuSyclNvidia;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagGpuSyclNvidia, TDim, TIdx>
        {
            using type = alpaka::AccGpuSyclNvidia<TDim, TIdx>;
        };
    } // namespace trait

} // namespace alpaka

#endif
