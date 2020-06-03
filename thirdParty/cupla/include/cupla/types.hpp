/* Copyright 2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdint>

#include "cupla/defines.hpp"
#include "cupla/namespace.hpp"

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

    using MemSizeType = size_t;
    using IdxType = unsigned int;

    static constexpr uint32_t Dimensions = 3u;

    template<
        uint32_t T_dim
    >
    using AlpakaDim = ::alpaka::dim::DimInt< T_dim >;

    using KernelDim = AlpakaDim< Dimensions >;

    using IdxVec3 = ::alpaka::vec::Vec<
        KernelDim,
        IdxType
    >;

    template<
        uint32_t T_dim
    >
    using MemVec = ::alpaka::vec::Vec<
        AlpakaDim< T_dim >,
        MemSizeType
    >;

    using AccHost = ::alpaka::dev::DevCpu;
    using AccHostStream = ::alpaka::queue::QueueCpuBlocking;

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED) ||                            \
    defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) ||                         \
    defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) ||                            \
    defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) ||                             \
    defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED) ||                             \
    defined(ALPAKA_ACC_CPU_BT_OMP4_ENABLED)

    using AccDev = ::alpaka::dev::DevCpu;
#   if (CUPLA_STREAM_ASYNC_ENABLED == 1)
        using AccStream = ::alpaka::queue::QueueCpuNonBlocking;
#   else
        using AccStream = ::alpaka::queue::QueueCpuBlocking;
#   endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    using Acc = ::alpaka::acc::AccCpuOmp2Threads<
        KernelDim,
        IdxType
    >;
#endif

#if (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED == 1)
    #if (CUPLA_NUM_SELECTED_DEVICES == 1)
        using Acc = ::alpaka::acc::AccCpuOmp2Blocks<
            KernelDim,
            IdxType
        >;
    #else
        using AccThreadSeq = ::alpaka::acc::AccCpuOmp2Blocks<
            KernelDim,
            IdxType
        >;
    #endif
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    using Acc = ::alpaka::acc::AccCpuThreads<
        KernelDim,
        IdxType
    >;
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    #if (CUPLA_NUM_SELECTED_DEVICES == 1)
        using Acc = ::alpaka::acc::AccCpuSerial<
            KernelDim,
            IdxType
        >;
    #else
        using AccThreadSeq = ::alpaka::acc::AccCpuSerial<
            KernelDim,
            IdxType
        >;
    #endif
#endif

#if (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED == 1)
    #if (CUPLA_NUM_SELECTED_DEVICES == 1)
        using Acc = ::alpaka::acc::AccCpuTbbBlocks<
            KernelDim,
            IdxType
        >;
    #else
        using AccThreadSeq = ::alpaka::acc::AccCpuTbbBlocks<
            KernelDim,
            IdxType
        >;
    #endif
#endif

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
    using Acc = ::alpaka::acc::AccCpuOmp4<
        KernelDim,
        IdxType
    >;
#endif

#endif


#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using AccDev = ::alpaka::dev::DevCudaRt;
#   if (CUPLA_STREAM_ASYNC_ENABLED == 1)
        using AccStream = ::alpaka::queue::QueueCudaRtNonBlocking;
#   else
        using AccStream = ::alpaka::queue::QueueCudaRtBlocking;
#   endif
    using Acc = ::alpaka::acc::AccGpuCudaRt<
        KernelDim,
        IdxType
    >;
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    using AccDev = ::alpaka::dev::DevHipRt;
#   if (CUPLA_STREAM_ASYNC_ENABLED == 1)
        using AccStream = ::alpaka::queue::QueueHipRtNonBlocking;
#   else
        using AccStream = ::alpaka::queue::QueueHipRtBlocking;
#   endif
    using Acc = ::alpaka::acc::AccGpuHipRt<
        KernelDim,
        IdxType
    >;
#endif

#if (CUPLA_NUM_SELECTED_DEVICES == 1)
    /** is an Alpaka accelerator which limits the thread count per block to one
     *
     * if only one accelerator is selected than it can be a accelerator without
     * thread restrictions
     */
    using AccThreadSeq = Acc;
#endif

    template<
        uint32_t T_dim
    >
    using AccBuf = ::alpaka::mem::buf::Buf<
        AccDev,
        uint8_t,
        AlpakaDim< T_dim >,
        MemSizeType
    >;

    template<
        uint32_t T_dim
    >
    using HostBuf = ::alpaka::mem::buf::Buf<
        AccHost,
        uint8_t,
        AlpakaDim< T_dim >,
        MemSizeType
    >;

    template<
        unsigned T_dim
    >
    using HostBufWrapper =
        ::alpaka::mem::view::ViewPlainPtr<
            AccHost,
            uint8_t,
            AlpakaDim< T_dim >,
            MemSizeType
        >;

    template<
        unsigned T_dim
    >
    using HostViewWrapper =
        ::alpaka::mem::view::ViewSubView<
            AccHost,
            uint8_t,
            AlpakaDim< T_dim >,
            MemSizeType
        >;

    template<
        unsigned T_dim
    >
    using DeviceBufWrapper =
        ::alpaka::mem::view::ViewPlainPtr<
            AccDev,
            uint8_t,
            AlpakaDim< T_dim >,
            MemSizeType
        >;

    template<
        unsigned T_dim
    >
    using DeviceViewWrapper =
        ::alpaka::mem::view::ViewSubView<
            AccDev,
            uint8_t,
            AlpakaDim< T_dim >,
            MemSizeType
        >;

} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namepsace cupla
