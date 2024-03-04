/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiHipRt.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka
{
    //! The HIP RT device handle.
    using DevHipRt = DevUniformCudaHipRt<ApiHipRt>;
} // namespace alpaka

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
