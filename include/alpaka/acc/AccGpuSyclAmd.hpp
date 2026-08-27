/* Copyright 2025 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Sycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU_AMD)

namespace alpaka
{
    //! The Amd GPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Amd GPU target device.
    template<typename TDim, typename TIdx>
    using AccGpuSyclAmd = AccGenericSycl<TagGpuSyclAmd, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccGpuSyclAmd<TDim, TIdx>>
        {
            using type = alpaka::TagGpuSyclAmd;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagGpuSyclAmd, TDim, TIdx>
        {
            using type = alpaka::AccGpuSyclAmd<TDim, TIdx>;
        };
    } // namespace trait

} // namespace alpaka

#endif
