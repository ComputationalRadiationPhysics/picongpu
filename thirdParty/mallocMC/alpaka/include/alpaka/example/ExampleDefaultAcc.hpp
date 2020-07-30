/* Copyright 2020 Jeffrey Kelling
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

#include <alpaka/alpaka.hpp>

#pragma once

namespace alpaka
{
    namespace example
    {
        //! Alias for the default accelerator used by examples. From a list of
        //! all accelerators the first one which is enabled is chosen.
        //! AccCpuSerial is selected last.
        template<class TDim, class TIdx>
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        using ExampleDefaultAcc = alpaka::acc::AccGpuCudaRt<TDim,TIdx>;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        using ExampleDefaultAcc = alpaka::acc::AccGpuHipRt<TDim,TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
        using ExampleDefaultAcc = alpaka::acc::AccCpuOmp2Blocks<TDim,TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
        using ExampleDefaultAcc = alpaka::acc::AccCpuTbbBlocks<TDim,TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
        using ExampleDefaultAcc = alpaka::acc::AccCpuFibers<TDim,TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
        using ExampleDefaultAcc = alpaka::acc::AccCpuOmp2Threads<TDim,TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
        using ExampleDefaultAcc = alpaka::acc::AccCpuThreads<TDim,TIdx>;
#elif defined(ALPAKA_ACC_CPU_BT_OMP4_ENABLED)
        using ExampleDefaultAcc = alpaka::acc::AccCpuOmp4<TDim,TIdx>;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
        using ExampleDefaultAcc = alpaka::acc::AccCpuSerial<TDim,TIdx>;
#else
        class ExampleDefaultAcc;
        #warning "No supported backend selected."
#endif
    }
}
