/**
 * Copyright 2016 Rene Widera
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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#   undef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#   define ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED 1
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#   undef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#   define ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED 1
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#   undef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#   define ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED 1
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#   undef ALPAKA_ACC_GPU_CUDA_ENABLED
#   define ALPAKA_ACC_GPU_CUDA_ENABLED 1
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#   undef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#   define ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED 1
#endif

#define CUPLA_NUM_SELECTED_DEVICES (                                           \
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED +                                  \
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED +                               \
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED  +                                 \
        ALPAKA_ACC_GPU_CUDA_ENABLED +                                          \
        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED                                     \
)


#if( CUPLA_NUM_SELECTED_DEVICES == 0 )
    #error "there is no accelerator selected, please run `ccmake .` and select one"
#endif

#if( CUPLA_NUM_SELECTED_DEVICES > 2  )
    #error "please select at most two accelerators"
#endif

// count accelerators where the thread count must be one
#define CUPLA_NUM_SELECTED_THREAD_SEQ_DEVICES (                                \
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED +                                  \
        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED                                     \
)

#define CUPLA_NUM_SELECTED_THREAD_PARALLEL_DEVICES (                           \
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED +                                  \
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED +                               \
        ALPAKA_ACC_GPU_CUDA_ENABLED                                            \
)

#if( CUPLA_NUM_SELECTED_THREAD_SEQ_DEVICES > 1 )
    #error "it is only alowed to select one thread sequential Alpaka accelerator"
#endif

#if( CUPLA_NUM_SELECTED_THREAD_PARALLEL_DEVICES > 1 )
    #error "it is only alowed to select one thread parallelized Alpaka accelerator"
#endif


namespace cupla {


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
    using AccHostStream = ::alpaka::stream::StreamCpuSync;

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED) ||                            \
    defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) ||                         \
    defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) ||                            \
    defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

    using AccDev = ::alpaka::dev::DevCpu;
#   if (CUPLA_STREAM_ASYNC_ENABLED == 1)
        using AccStream = ::alpaka::stream::StreamCpuAsync;
#   else
        using AccStream = ::alpaka::stream::StreamCpuSync;
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

#endif


#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using AccDev = ::alpaka::dev::DevCudaRt;
#   if (CUPLA_STREAM_ASYNC_ENABLED == 1)
        using AccStream = ::alpaka::stream::StreamCudaRtAsync;
#   else
        using AccStream = ::alpaka::stream::StreamCudaRtSync;
#   endif
    using Acc = ::alpaka::acc::AccGpuCudaRt<
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
} // namepsace cupla

