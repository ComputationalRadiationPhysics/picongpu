/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/sync/Traits.hpp"
#include "alpaka/core/Config.hpp"
#include "alpaka/core/Interface.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    //! The GPU CUDA/HIP block synchronization.
    class BlockSyncUniformCudaHipBuiltIn
        : public interface::Implements<ConceptBlockSync, BlockSyncUniformCudaHipBuiltIn>
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !ALPAKA_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !ALPAKA_LANG_HIP
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
#        if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == 0 && ALPAKA_COMP_HIP
#            if ALPAKA_COMP_CLANG >= ALPAKA_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wunique-object-duplication"
#            endif
                // workaround for unsupported syncthreads_* operation on AMD hardware without sync extension
                __shared__ int tmp;
#            if ALPAKA_COMP_CLANG >= ALPAKA_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic pop
#            endif
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
#        if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == 0 && ALPAKA_COMP_HIP
#            if ALPAKA_COMP_CLANG >= ALPAKA_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wunique-object-duplication"
#            endif
                // workaround for unsupported syncthreads_* operation on AMD hardware without sync extension
                __shared__ int tmp;
#            if ALPAKA_COMP_CLANG >= ALPAKA_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic pop
#            endif
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
#        if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__ == 0 && ALPAKA_COMP_HIP
#            if ALPAKA_COMP_CLANG >= ALPAKA_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic push
#                pragma clang diagnostic ignored "-Wunique-object-duplication"
#            endif
                // workaround for unsupported syncthreads_* operation on AMD hardware without sync extension
                __shared__ int tmp;
#            if ALPAKA_COMP_CLANG >= ALPAKA_VERSION_NUMBER(21, 0, 0)
#                pragma clang diagnostic pop
#            endif
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
