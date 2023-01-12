/* Copyright 2022 Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#    include <alpaka/acc/Tag.hpp>
#    include <alpaka/core/ApiCudaRt.hpp>

namespace alpaka
{
    template<typename TDim, typename TIdx>
    using AccGpuCudaRt = AccGpuUniformCudaHipRt<ApiCudaRt, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccGpuCudaRt<TDim, TIdx>>
        {
            using type = alpaka::TagGpuCudaRt;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagGpuCudaRt, TDim, TIdx>
        {
            using type = alpaka::AccGpuCudaRt<TDim, TIdx>;
        };
    } // namespace trait
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
