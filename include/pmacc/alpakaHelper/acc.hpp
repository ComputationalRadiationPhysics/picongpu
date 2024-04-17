/* Copyright 2024 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace pmacc
{
    using IdxType = uint32_t;
    using MemIdxType = size_t;

    template<uint32_t T_dim>
    using AlpakaDim = ::alpaka::DimInt<T_dim>;
    using HostDevice = ::alpaka::DevCpu;

#if(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using ComputeDevice = ::alpaka::DevCudaRt;
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccGpuCudaRt<AlpakaDim<T_dim>, IdxType>;
#elif(ALPAKA_ACC_GPU_HIP_ENABLED)
    using ComputeDevice = ::alpaka::DevHipRt;
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccGpuHipRt<AlpakaDim<T_dim>, IdxType>;
#elif(                                                                                                                \
    ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED || ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED                                     \
    || ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED                                      \
    || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)

    using ComputeDevice = ::alpaka::DevCpu;

#    if(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuOmp2Threads<AlpakaDim<T_dim>, IdxType>;
#    endif

#    if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuOmp2Blocks<AlpakaDim<T_dim>, IdxType>;
#    endif

#    if(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuThreads<AlpakaDim<T_dim>, IdxType>;
#    endif

#    if(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuSerial<AlpakaDim<T_dim>, IdxType>;
#    endif

#    if(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuTbbBlocks<AlpakaDim<T_dim>, IdxType>;
#    endif
#endif

#if(PMACC_USE_ASYNC_QUEUES == 1)
    using ComputeQueue = ::alpaka::Queue<ComputeDevice, ::alpaka::NonBlocking>;
#else
    using ComputeQueue = ::alpaka::Queue<ComputeDevice, ::alpaka::Blocking>;
#endif

    using ComputeEvent = alpaka::Event<ComputeQueue>;

    /*! device compile flag
     *
     * Enabled if the compiler processes currently a separate compile path for the device code
     *
     * @attention value is always 0 for alpaka CPU accelerators
     *
     * Value is 1 if device path is compiled else 0
     */
#if defined(__CUDA_ARCH__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1 && defined(__HIP__))
#    define PMACC_DEVICE_COMPILE 1
#else
#    define PMACC_DEVICE_COMPILE 0
#endif

} // namespace pmacc
