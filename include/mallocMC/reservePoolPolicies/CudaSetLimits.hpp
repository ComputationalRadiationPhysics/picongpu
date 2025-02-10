/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2014-2024 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian Lenz - j.lenz ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include "CudaSetLimits.hpp"

#    include <cuda_runtime_api.h>

#    include <mutex>
#    include <string>

namespace mallocMC
{
    namespace ReservePoolPolicies
    {
        /**
         * @brief set CUDA internal heap for device-side malloc calls
         *
         * This ReservePoolPolicy is intended for use with CUDA capable
         * accelerators that support at least compute capability 2.0. It should
         * be used in conjunction with a CreationPolicy that actually requires
         * the CUDA-internal heap to be sized by calls to cudaDeviceSetLimit().
         *
         * This policy sets the cudaLimitMallocHeapSize device limit. This value
         * can no longer be changed once a kernel using ::malloc()/::free() has
         * been run. Subsequent attempts will result in errors unless the device
         * is reset via cudaDeviceReset(). See:
         * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g05956f16eaa47ef3a4efee84563ccb7d
         */
        // TODO alpaka
        struct CudaSetLimits
        {
            template<typename AlpakaDev>
            auto setMemPool(AlpakaDev const& dev, size_t memsize) -> void*
            {
                cudaDeviceSetLimit(cudaLimitMallocHeapSize, memsize);
                return nullptr;
            }

            static void resetMemPool()
            {
                cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8192U);
                cudaGetLastError(); // cudaDeviceSetLimit() usually fails if any
                                    // kernel before used ::malloc(), so let's
                                    // clear the error state
            }

            static auto classname() -> std::string
            {
                return "CudaSetLimits";
            }
        };

    } // namespace ReservePoolPolicies
} // namespace mallocMC

#endif
