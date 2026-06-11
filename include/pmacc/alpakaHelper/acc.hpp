/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

#if (ALPAKA_ACC_GPU_CUDA_ENABLED)
    using ComputeDevice = ::alpaka::DevCudaRt;
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccGpuCudaRt<AlpakaDim<T_dim>, IdxType>;
#elif (ALPAKA_ACC_GPU_HIP_ENABLED)
    using ComputeDevice = ::alpaka::DevHipRt;
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccGpuHipRt<AlpakaDim<T_dim>, IdxType>;
#elif (                                                                                                               \
    ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED || ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED                                     \
    || ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED                                      \
    || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)

    using ComputeDevice = ::alpaka::DevCpu;

#    if (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuOmp2Threads<AlpakaDim<T_dim>, IdxType>;
#    endif

#    if (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuOmp2Blocks<AlpakaDim<T_dim>, IdxType>;
#    endif

#    if (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuThreads<AlpakaDim<T_dim>, IdxType>;
#    endif

#    if (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuSerial<AlpakaDim<T_dim>, IdxType>;
#    endif

#    if (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    template<uint32_t T_dim>
    using Acc = ::alpaka::AccCpuTbbBlocks<AlpakaDim<T_dim>, IdxType>;
#    endif
#endif

#if (PMACC_USE_ASYNC_QUEUES == 1)
    using ComputeDeviceQueue = ::alpaka::Queue<ComputeDevice, ::alpaka::NonBlocking>;
#else
    using ComputeDeviceQueue = ::alpaka::Queue<ComputeDevice, ::alpaka::Blocking>;
#endif

    using ComputeDeviceEvent = alpaka::Event<ComputeDeviceQueue>;

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
