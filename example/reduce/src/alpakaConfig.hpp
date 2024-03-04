/* Copyright 2023 Jonas Schenke, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "iterator.hpp"

#include <alpaka/alpaka.hpp>

// Defines for dimensions and types.
using Dim = alpaka::DimInt<1u>;
using Idx = uint64_t;
using Extent = uint64_t;
using WorkDiv = alpaka::WorkDivMembers<Dim, Extent>;

//! Returns the supplied number or the maxumim number of threads per block for a
//! specific accelerator.
//!
//! \tparam TAcc The accelerator object.
//! \tparam TSize The desired size.
template<typename TAcc, uint64_t TSize>
static constexpr auto getMaxBlockSize() -> uint64_t
{
    return (TAcc::MaxBlockSize::value > TSize) ? TSize : TAcc::MaxBlockSize::value;
}

//! Get Trait via struct.
//!
//! \tparam T The data type.
//! \tparam TBuf The buffer type.
//! \tparam TAcc The accelerator type.
//!
//! Defines the appropriate iterator for an accelerator.
template<typename T, typename TBuf, typename TAcc>
struct GetIterator
{
    using Iterator = IteratorCpu<TAcc, T, TBuf>;
};

// Note: OpenMP 2 Threads and TBB Blocks accelerators aren't implented

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
//! OpenMP 2 Blocks defines
//!
//! Defines Host, Device, etc. for the OpenMP 2 Blocks accelerator.
struct CpuOmp2Blocks
{
    using Host = alpaka::AccCpuOmp2Blocks<Dim, Extent>;
    using Acc = alpaka::AccCpuOmp2Blocks<Dim, Extent>;
    using SmCount = alpaka::DimInt<1u>;
    using MaxBlockSize = alpaka::DimInt<1u>;
};

template<typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuOmp2Blocks<TArgs...>>
{
    using Iterator = IteratorCpu<alpaka::AccCpuOmp2Blocks<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//! Serial CPU defines
//!
//! Defines Host, Device, etc. for the serial CPU accelerator.
struct CpuSerial
{
    using Host = alpaka::AccCpuSerial<Dim, Extent>;
    using Acc = alpaka::AccCpuSerial<Dim, Extent>;
    using MaxBlockSize = alpaka::DimInt<1u>;
};

template<typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuSerial<TArgs...>>
{
    using Iterator = IteratorCpu<alpaka::AccCpuSerial<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
//! CPU Threads defines
//!
//! Defines Host, Device, etc. for the CPU Threads accelerator.
struct CpuThreads
{
    using Host = alpaka::AccCpuThreads<Dim, Extent>;
    using Acc = alpaka::AccCpuThreads<Dim, Extent>;
    using MaxBlockSize = alpaka::DimInt<1u>;
};

template<typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccCpuThreads<TArgs...>>
{
    using Iterator = IteratorCpu<alpaka::AccCpuThreads<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#    ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//! CUDA defines
//!
//! Defines Host, Device, etc. for the CUDA/HIP accelerator.
struct GpuCudaRt
{
    using Host = alpaka::AccCpuSerial<Dim, Extent>;
    using Acc = alpaka::AccGpuCudaRt<Dim, Extent>;
    using MaxBlockSize = alpaka::DimInt<1024u>;
};

template<typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::AccGpuUniformCudaHipRt<TArgs...>>
{
    using Iterator = IteratorGpu<alpaka::AccGpuUniformCudaHipRt<TArgs...>, T, TBuf>;
};
#    endif
#endif
