/**
 * \file
 * Copyright 2018 Sebastian Benner, Jonas Schenke
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 *
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
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Stream = alpaka::queue::QueueCpuSync;
    using Event = alpaka::event::Event<Stream>;
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
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Stream = alpaka::queue::QueueCpuSync;
    using Event = alpaka::event::Event<Stream>;
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
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Stream = alpaka::queue::QueueCpuSync;
    using Event = alpaka::event::Event<Stream>;
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
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Stream = alpaka::queue::QueueCpuSync;
    using Event = alpaka::event::Event<Stream>;
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
//! Defines Host, Device, etc. for the CUDA accelerator.
struct GpuCudaRt
{
    using Host = alpaka::acc::AccCpuSerial<Dim, Extent>;
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Extent>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Stream = alpaka::queue::QueueCudaRtAsync;
    using Event = alpaka::event::Event<Stream>;
    using MaxBlockSize = alpaka::dim::DimInt<1024u>;
};

template <typename T, typename TBuf, typename... TArgs>
struct GetIterator<T, TBuf, alpaka::acc::AccGpuCudaRt<TArgs...>>
{
    using Iterator = IteratorGpu<alpaka::acc::AccGpuCudaRt<TArgs...>, T, TBuf>;
};
#endif
#endif
