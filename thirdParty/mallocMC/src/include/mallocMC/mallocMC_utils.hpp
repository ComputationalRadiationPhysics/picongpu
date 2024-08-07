/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp
  https://www.hzdr.de/crp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at
              Carlchristian Eckert - c.eckert ( at ) hzdr.de

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

#ifdef _MSC_VER
#    include <intrin.h>
#endif

#include <atomic>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
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
    template<int PSIZE>
    class __PointerEquivalent
    {
    public:
        using type = unsigned int;
    };
    template<>
    class __PointerEquivalent<8>
    {
    public:
        using type = unsigned long long;
    };

#if defined(__CUDA_ARCH__)
    constexpr auto warpSize = 32; // TODO
#elif(MALLOCMC_DEVICE_COMPILE && BOOST_COMP_HIP)
// defined:
// https://github.com/llvm/llvm-project/blob/62ec4ac90738a5f2d209ed28c822223e58aaaeb7/clang/lib/Basic/Targets/AMDGPU.cpp#L400
// overview wave front size:
// https://github.com/llvm/llvm-project/blob/efc063b621ea0c4d1e452bcade62f7fc7e1cc937/clang/test/Driver/amdgpu-macros.cl#L70-L115
// gfx10XX has 32 threads per wavefront else 64
#    if(HIP_VERSION_MAJOR >= 4)
    constexpr auto warpSize = __AMDGCN_WAVEFRONT_SIZE;
#    else
    constexpr auto warpSize = 64;
#    endif
#else
    constexpr auto warpSize = 1;
#endif

    using PointerEquivalent = mallocMC::__PointerEquivalent<sizeof(char*)>::type;

    ALPAKA_FN_ACC inline auto laneid()
    {
#if defined(__CUDA_ARCH__)
        std::uint32_t mylaneid;
        asm("mov.u32 %0, %%laneid;" : "=r"(mylaneid));
        return mylaneid;
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP__)
        return __lane_id();
#else
        return 0u;
#endif
    }

    /** warp index within a multiprocessor
     *
     * Index of the warp within the multiprocessor at the moment of the query.
     * The result is volatile and can be different with each query.
     *
     * @return current index of the warp
     */
    ALPAKA_FN_ACC inline auto warpid()
    {
#if defined(__CUDA_ARCH__)
        std::uint32_t mywarpid;
        asm("mov.u32 %0, %%warpid;" : "=r"(mywarpid));
        return mywarpid;
#elif(MALLOCMC_DEVICE_COMPILE && BOOST_COMP_HIP)
        // get wave id
        // https://github.com/ROCm-Developer-Tools/HIP/blob/f72a669487dd352e45321c4b3038f8fe2365c236/include/hip/hcc_detail/device_functions.h#L974-L1024
        return __builtin_amdgcn_s_getreg(GETREG_IMMED(3, 0, 4));
#else
        return 0u;
#endif
    }

    ALPAKA_FN_ACC inline auto smid()
    {
#if defined(__CUDA_ARCH__)
        std::uint32_t mysmid;
        asm("mov.u32 %0, %%smid;" : "=r"(mysmid));
        return mysmid;
#elif(MALLOCMC_DEVICE_COMPILE && BOOST_COMP_HIP)
        return __smid();
#else
        return 0u;
#endif
    }

    ALPAKA_FN_ACC inline auto lanemask_lt()
    {
#if defined(__CUDA_ARCH__)
        std::uint32_t lanemask;
        asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask));
        return lanemask;
#elif(MALLOCMC_DEVICE_COMPILE && BOOST_COMP_HIP)
        return __lanemask_lt();
#else
        return 0u;
#endif
    }

    ALPAKA_FN_ACC inline auto ballot(int pred)
    {
#if defined(__CUDA_ARCH__)
        return __ballot_sync(__activemask(), pred);
#elif(MALLOCMC_DEVICE_COMPILE && BOOST_COMP_HIP)
        // return value is 64bit for HIP-clang
        return __ballot(pred);
#else
        return 1u;
#endif
    }


    ALPAKA_FN_ACC inline auto activemask()
    {
#if defined(__CUDA_ARCH__)
        return __activemask();
#elif(MALLOCMC_DEVICE_COMPILE && BOOST_COMP_HIP)
        // return value is 64bit for HIP-clang
        return ballot(1);
#else
        return 1u;
#endif
    }

    template<class T>
    ALPAKA_FN_HOST_ACC inline auto divup(T a, T b) -> T
    {
        return (a + b - 1) / b;
    }

    /** the maximal number threads per block, valid for sm_2.X - sm_7.5
     *
     * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
     */
    constexpr uint32_t maxThreadsPerBlock = 1024;

    /** warp id within a cuda block
     *
     * The id is constant over the lifetime of the thread.
     * The id is not equal to warpid().
     *
     * @return warp id within the block
     */
    template<typename AlpakaAcc>
    ALPAKA_FN_ACC inline auto warpid_withinblock(const AlpakaAcc& acc) -> std::uint32_t
    {
        const auto localId = alpaka::mapIdx<1>(
            alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc),
            alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc))[0];
        return localId / warpSize;
    }

    template<typename T>
    ALPAKA_FN_ACC inline auto ffs(T mask) -> std::uint32_t
    {
#if defined(__CUDA_ARCH__)
        return ::__ffs(mask);
#elif(MALLOCMC_DEVICE_COMPILE && BOOST_COMP_HIP)
        // return value is 64bit for HIP-clang
        return ::__ffsll(static_cast<unsigned long long int>(mask));
#else
        if(mask == 0)
            return 0;
        auto i = 1u;
        while((mask & 1) == 0)
        {
            mask >>= 1;
            i++;
        }
        return i;
#endif
    }

    template<typename T>
    ALPAKA_FN_ACC inline auto popc(T mask) -> std::uint32_t
    {
#if defined(__CUDA_ARCH__)
        return ::__popc(mask);
#elif(MALLOCMC_DEVICE_COMPILE && BOOST_COMP_HIP)
        // return value is 64bit for HIP-clang
        return ::__popcll(static_cast<unsigned long long int>(mask));
#else
        // cf.
        // https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
        std::uint32_t count = 0;
        while(mask)
        {
            count++;
            mask &= mask - 1;
        }
        return count;
#endif
    }

    // Threadfence implementations will maybe moved later into alpaka
    template<typename T_Acc, typename T_Sfinae = void>
    struct ThreadFence
    {
        // CPU only implementation
        static void device()
        {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }

        static void block()
        {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
    };

    template<typename... T_AccArgs>
    struct ThreadFence<alpaka::AccGpuUniformCudaHipRt<T_AccArgs...>, void>
    {
        static ALPAKA_FN_ACC void device()
        {
#if MALLOCMC_DEVICE_COMPILE
            __threadfence();
#endif
        }

        static ALPAKA_FN_ACC void block()
        {
#if MALLOCMC_DEVICE_COMPILE
            __threadfence_block();
#endif
        }
    };

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T_Acc>
    ALPAKA_FN_ACC void threadfenceDevice(T_Acc const& acc)
    {
        ThreadFence<T_Acc>::device();
    }

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T_Acc>
    ALPAKA_FN_ACC void threadfenceBlock(T_Acc const& acc)
    {
        ThreadFence<T_Acc>::block();
    }
} // namespace mallocMC
