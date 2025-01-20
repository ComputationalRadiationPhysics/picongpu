/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp
  https://www.hzdr.de/crp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014-2024 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at
              Carlchristian Eckert - c.eckert ( at ) hzdr.de
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

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Common.hpp>

#include <sys/types.h>

#ifdef _MSC_VER
#    include <intrin.h>
#endif

#include <cstdint>
#include <type_traits>

/* HIP-clang is doing something wrong and uses the host path of the code when __HIP_DEVICE_COMPILE__
 * only is used to detect the device compile path.
 * Since we require devices with support for ballot we can high-jack __HIP_ARCH_HAS_WARP_BALLOT__.
 */
#if(defined(__HIP_ARCH_HAS_WARP_BALLOT__) || defined(__CUDA_ARCH__) || __HIP_DEVICE_COMPILE__ == 1)
#    define MALLOCMC_DEVICE_COMPILE 1
#endif

namespace mallocMC
{

    template<typename TAcc>
    constexpr uint32_t warpSize = 1U;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    constexpr uint32_t warpSize<alpaka::AccGpuCudaRt<TDim, TIdx>> = 32U;
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#    if(HIP_VERSION_MAJOR >= 4)
    template<typename TDim, typename TIdx>
    constexpr uint32_t warpSize<alpaka::AccGpuHipRt<TDim, TIdx>> = __AMDGCN_WAVEFRONT_SIZE;
#    else
    template<typename TDim, typename TIdx>
    constexpr uint32_t warpSize<alpaka::AccGpuHipRt<TDim, TIdx>> = 64;
#    endif
#endif

    ALPAKA_FN_ACC inline auto laneid()
    {
#if defined(__CUDA_ARCH__)
        std::uint32_t mylaneid;
        asm("mov.u32 %0, %%laneid;" : "=r"(mylaneid));
        return mylaneid;
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP__)
        return __lane_id();
#else
        return 0U;
#endif
    }

    /** warp index within a multiprocessor
     *
     * Index of the warp within the multiprocessor at the moment of the query.
     * The result is volatile and can be different with each query.
     *
     * @return current index of the warp
     */
    template<typename TAcc>
    ALPAKA_FN_ACC inline auto warpid(TAcc const& /*acc*/) -> uint32_t
    {
        return 0U;
    }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    // ALPAKA_FN_ACC resolves to `__host__ __device__` if we're not in CUDA_ONLY_MODE. But the assembly instruction is
    // specific to the device and cannot be compiled on the host. So, we need an explicit `__device__` here.`
    inline __device__ auto warpid(alpaka::AccGpuCudaRt<TDim, TIdx> const& /*acc*/) -> uint32_t
    {
        std::uint32_t mywarpid = 0;
        asm("mov.u32 %0, %%warpid;" : "=r"(mywarpid));
        return mywarpid;
    }
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    template<typename TDim, typename TIdx>
    ALPAKA_FN_ACC inline auto warpid(alpaka::AccGpuHipRt<TDim, TIdx> const& /*acc*/) -> uint32_t
    {
        // get wave id
        // https://github.com/ROCm-Developer-Tools/HIP/blob/f72a669487dd352e45321c4b3038f8fe2365c236/include/hip/hcc_detail/device_functions.h#L974-L1024
        return __builtin_amdgcn_s_getreg(GETREG_IMMED(3, 0, 4));
    }
#endif

    template<typename TAcc>
    ALPAKA_FN_ACC inline auto smid(TAcc const& /*acc*/) -> uint32_t
    {
        return 0U;
    }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    ALPAKA_FN_ACC inline auto smid(alpaka::AccGpuCudaRt<TDim, TIdx> const& /*acc*/) -> uint32_t
    {
        std::uint32_t mysmid = 0;
        asm("mov.u32 %0, %%smid;" : "=r"(mysmid));
        return mysmid;
    }
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    template<typename TDim, typename TIdx>
    ALPAKA_FN_ACC inline auto smid(alpaka::AccGpuHipRt<TDim, TIdx> const& /*acc*/) -> uint32_t
    {
        return __smid();
    }
#endif

    template<typename TAcc>
    ALPAKA_FN_ACC inline auto lanemask_lt(TAcc const& /*acc*/)
    {
        return 0U;
    }
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    ALPAKA_FN_ACC inline auto lanemask_lt(alpaka::AccGpuCudaRt<TDim, TIdx> const& /*acc*/)
    {
        std::uint32_t lanemask;
        asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask));
        return lanemask;
    }
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    template<typename TDim, typename TIdx>
    ALPAKA_FN_ACC inline auto lanemask_lt(alpaka::AccGpuHipRt<TDim, TIdx> const& /*acc*/)
    {
        return __lanemask_lt();
    }
#endif


    /** the maximal number threads per block, valid for sm_2.X - sm_7.5
     *
     * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
     */
    constexpr uint32_t maxThreadsPerBlock = 1024U;

    /** warp id within a cuda block
     *
     * The id is constant over the lifetime of the thread.
     * The id is not equal to warpid().
     *
     * @return warp id within the block
     */
    template<typename AlpakaAcc>
    ALPAKA_FN_ACC inline auto warpid_withinblock(AlpakaAcc const& acc) -> std::uint32_t
    {
        auto const localId = alpaka::mapIdx<1>(
            alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc),
            alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc))[0];
        return localId / warpSize<AlpakaAcc>;
    }

    template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto ceilingDivision(T const numerator, U const denominator) -> T
    {
        return (numerator + (denominator - 1)) / denominator;
    }

    template<typename T_size>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto indexOf(
        void const* const pointer,
        void const* const start,
        T_size const stepSize) -> std::make_signed_t<T_size>
    {
        return std::distance(reinterpret_cast<char const*>(start), reinterpret_cast<char const*>(pointer)) / stepSize;
    }

    template<typename TAcc, typename T>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto atomicLoad(TAcc const& acc, T& target)
    {
        return alpaka::atomicCas(acc, &target, static_cast<T>(0U), static_cast<T>(0U));
    }
} // namespace mallocMC
