/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGpuUniformCudaHipRt.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/ApiHipRt.hpp"

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka
{
    template<typename TDim, typename TIdx>
    using AccGpuHipRt = AccGpuUniformCudaHipRt<ApiHipRt, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccGpuHipRt<TDim, TIdx>>
        {
            using type = alpaka::TagGpuHipRt;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagGpuHipRt, TDim, TIdx>
        {
            using type = alpaka::AccGpuHipRt<TDim, TIdx>;
        };
    } // namespace trait
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
