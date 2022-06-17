/* Copyright 2022 Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#    include <alpaka/core/ApiHipRt.hpp>

namespace alpaka
{
    template<typename TDim, typename TIdx>
    using AccGpuHipRt = AccGpuUniformCudaHipRt<ApiHipRt, TDim, TIdx>;
}

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
