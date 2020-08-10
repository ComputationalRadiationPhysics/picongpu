/* Copyright 2019 Jonas Schenke
 *
 * This file exemplifies usage of alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#pragma once

#include "iterator.hpp"
#include <alpaka/alpaka.hpp>

// Defines for dimensions and types.
using Dim = alpaka::dim::DimInt<1u>;
using Idx = uint64_t;
using Extent = uint64_t;
using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Extent>;

//-----------------------------------------------------------------------------
//! Returns the supplied number or the maxumim number of threads per block for a
//! specific accelerator.
//!
//! \tparam TAcc The accelerator object.
//! \tparam TSize The desired size.
template <typename TAcc, uint64_t TSize>
static constexpr uint64_t getMaxBlockSize()
{
    return (TAcc::MaxBlockSize::value > TSize) ? TSize
                                               : TAcc::MaxBlockSize::value;
}

//#############################################################################
//! Get Trait via struct.
//!
//! \tparam T The data type.
//! \tparam TBuf The buffer type.
//! \tparam TAcc The accelerator type.
//!
//! Defines the appropriate iterator for an accelerator.
template <typename T, typename TBuf, typename TAcc>
struct GetIterator
{
    using Iterator = IteratorCpu<TAcc, T, TBuf>;
};

// Note: Boost Fibers, OpenMP 2 Threads and TBB Blocks accelerators aren't implented

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
//#############################################################################
//! OpenMP 2 Blocks defines
//!
//! Defines Host, Device, etc. for the OpenMP 2 Blocks accelerator.
struct CpuOmp2Blocks
{
    using Host = alpaka::acc::AccCpuOmp2Blocks<Dim, Extent>;
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Extent>;
    using SmCount = alpaka::dim::DimInt<1u>;
    using MaxBlockSize = alpaka::dim::DimInt<1u>;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccCpuOmp2Blocks<TArgs...>>
{
    using Iterator =
        IteratorCpu<alpaka::acc::AccCpuOmp2Blocks<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! OpenMP 4 defines
//!
//! Defines Host, Device, etc. for the OpenMP 4 accelerator.
struct CpuOmp4
{
    using Host = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using Acc = alpaka::acc::AccCpuOmp4<Dim, Extent>;
    using MaxBlockSize = alpaka::dim::DimInt<1u>;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccCpuOmp4<TArgs...>>
{
    using Iterator = IteratorCpu<alpaka::acc::AccCpuOmp4<TArgs...>, T, TBuf>;
};
#endif
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! Serial CPU defines
//!
//! Defines Host, Device, etc. for the serial CPU accelerator.
struct CpuSerial
{
    using Host = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using Acc = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using MaxBlockSize = alpaka::dim::DimInt<1u>;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccCpuSerial<TArgs...>>
{
    using Iterator = IteratorCpu<alpaka::acc::AccCpuSerial<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
//#############################################################################
//! CPU Threads defines
//!
//! Defines Host, Device, etc. for the CPU Threads accelerator.
struct CpuThreads
{
    using Host = alpaka::acc::AccCpuThreads<Dim, Extent>;
    using Acc = alpaka::acc::AccCpuThreads<Dim, Extent>;
    using MaxBlockSize = alpaka::dim::DimInt<1u>;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccCpuThreads<TArgs...>>
{
    using Iterator = IteratorCpu<alpaka::acc::AccCpuThreads<TArgs...>, T, TBuf>;
};
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//#############################################################################
//! CUDA defines
//!
//! Defines Host, Device, etc. for the CUDA/HIP accelerator.
struct GpuCudaRt
{
    using Host = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Extent>;
    using MaxBlockSize = alpaka::dim::DimInt<1024u>;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccGpuUniformCudaHipRt<TArgs...>>
{
    using Iterator = IteratorGpu<alpaka::acc::AccGpuUniformCudaHipRt<TArgs...>, T, TBuf>;
};
#endif
#endif
