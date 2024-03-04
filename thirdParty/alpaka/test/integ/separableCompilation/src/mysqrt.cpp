/* Copyright 2023 Benjamin Worpitz, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "mysqrt.hpp"

#if defined(__CUDA_ARCH__)
#    include <cuda_runtime.h>
#elif defined(__HIP_DEVICE_COMPILE__)
#    include <hip_runtime.h>
#elif defined(__SYCL_DEVICE_ONLY__)
#    include <sycl/sycl.hpp>
#else
#    include <cmath>
#endif

// Computes x * y + z as if to infinite precision and rounded only once to fit the result type.
inline ALPAKA_FN_HOST_ACC auto myfma(float x, float y, float z) noexcept -> float
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fmaf(x, y, z);
#elif defined(__SYCL_DEVICE_ONLY__)
    return sycl::fma(x, y, z);
#else
    return std::fma(x, y, z);
#endif
}

// A square root calculation using simple operations.
ALPAKA_FN_HOST_ACC auto mysqrt(float x) noexcept -> float
{
    if(x <= 0)
    {
        return 0.0f;
    }

    float result = x;

    for(int i = 0; i < 100; ++i)
    {
        if(result <= 0)
        {
            result = 0.1f;
        }
        float delta = myfma(-result, result, x);
        result = result + 0.5f * delta / result;
    }
    return result;
}
