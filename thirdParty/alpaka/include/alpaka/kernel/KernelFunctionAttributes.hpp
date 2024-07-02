/* Copyright 2022 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>

namespace alpaka
{
    //! Kernel function attributes struct. Attributes are filled by calling the API of the accelerator using the kernel
    //! function as an argument. In case of a CPU backend, maxThreadsPerBlock is set to 1 and other values remain zero
    //! since there are no correponding API functions to get the values.
    struct KernelFunctionAttributes
    {
        std::size_t constSizeBytes{0};
        std::size_t localSizeBytes{0};
        std::size_t sharedSizeBytes{0};
        int maxDynamicSharedSizeBytes{0};
        int numRegs{0};
        // This field is ptx or isa version if the backend is GPU
        int asmVersion{0};
        int maxThreadsPerBlock{0};
    };
} // namespace alpaka
