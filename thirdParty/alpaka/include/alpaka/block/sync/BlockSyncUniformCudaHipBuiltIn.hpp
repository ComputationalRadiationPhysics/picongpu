/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/sync/Traits.hpp"
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    //! The GPU CUDA/HIP block synchronization.
    class BlockSyncUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptBlockSync, BlockSyncUniformCudaHipBuiltIn>
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        template<>
        struct SyncBlockThreads<BlockSyncUniformCudaHipBuiltIn>
        {
            __device__ static auto syncBlockThreads(BlockSyncUniformCudaHipBuiltIn const& /*blockSync*/) -> void
            {
                __syncthreads();
            }
        };

        template<>
        struct SyncBlockThreadsPredicate<BlockCount, BlockSyncUniformCudaHipBuiltIn>
        {
            __device__ static auto syncBlockThreadsPredicate(
                BlockSyncUniformCudaHipBuiltIn const& /*blockSync*/,
                int predicate) -> int
            {
#        if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == 0 && BOOST_COMP_HIP
                // workaround for unsupported syncthreads_* operation on AMD hardware without sync extension
                __shared__ int tmp;
                __syncthreads();
                if(threadIdx.x == 0)
                    tmp = 0;
                __syncthreads();
                if(predicate)
                    ::atomicAdd(&tmp, 1);
                __syncthreads();

                return tmp;
#        else
                return __syncthreads_count(predicate);
#        endif
            }
        };

        template<>
        struct SyncBlockThreadsPredicate<BlockAnd, BlockSyncUniformCudaHipBuiltIn>
        {
            __device__ static auto syncBlockThreadsPredicate(
                BlockSyncUniformCudaHipBuiltIn const& /*blockSync*/,
                int predicate) -> int
            {
#        if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == 0 && BOOST_COMP_HIP
                // workaround for unsupported syncthreads_* operation on AMD hardware without sync extension
                __shared__ int tmp;
                __syncthreads();
                if(threadIdx.x == 0)
                    tmp = 1;
                __syncthreads();
                if(!predicate)
                    ::atomicAnd(&tmp, 0);
                __syncthreads();

                return tmp;
#        else
                return __syncthreads_and(predicate);
#        endif
            }
        };

        template<>
        struct SyncBlockThreadsPredicate<BlockOr, BlockSyncUniformCudaHipBuiltIn>
        {
            __device__ static auto syncBlockThreadsPredicate(
                BlockSyncUniformCudaHipBuiltIn const& /*blockSync*/,
                int predicate) -> int
            {
#        if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == 0 && BOOST_COMP_HIP
                // workaround for unsupported syncthreads_* operation on AMD hardware without sync extension
                __shared__ int tmp;
                __syncthreads();
                if(threadIdx.x == 0)
                    tmp = 0;
                __syncthreads();
                if(predicate)
                    ::atomicOr(&tmp, 1);
                __syncthreads();

                return tmp;
#        else
                return __syncthreads_or(predicate);
#        endif
            }
        };
    } // namespace trait

#    endif

} // namespace alpaka

#endif
