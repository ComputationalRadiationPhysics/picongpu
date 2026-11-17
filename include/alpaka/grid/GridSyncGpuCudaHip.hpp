/* Copyright 2026 Mykhailo Varvarin, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Config.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/grid/Traits.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
// The cooperative groups headers from CUDA 12.0.0 use PTX register names with a single '%' in an inline asm statement
// with operands, which clang rejects ("invalid % escape in inline assembly string"). The asm is only used for the
// multi-warp scratch space in the reserved shared memory, which the grid synchronization does not need, so disable it.
#        if ALPAKA_COMP_CLANG_CUDA && defined(CUDART_VERSION) && (CUDART_VERSION == 12000)                            \
            && !defined(_CG_USER_PROVIDED_SHARED_MEMORY)
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#            define _CG_USER_PROVIDED_SHARED_MEMORY
#            pragma clang diagnostic pop
#        endif
#        include <cooperative_groups.h>
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#        include <hip/hip_cooperative_groups.h>
#    endif


namespace alpaka
{
    //! The GPU CUDA/HIP grid synchronization.
    class GridSyncCudaHipBuiltIn : public interface::Implements<ConceptGridSync, GridSyncCudaHipBuiltIn>
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
        struct SyncGridThreads<GridSyncCudaHipBuiltIn>
        {
            __device__ static auto syncGridThreads(GridSyncCudaHipBuiltIn const& /*gridSync*/) -> void
            {
                cooperative_groups::this_grid().sync();
            }
        };

    } // namespace trait

#    endif

} // namespace alpaka

#endif
