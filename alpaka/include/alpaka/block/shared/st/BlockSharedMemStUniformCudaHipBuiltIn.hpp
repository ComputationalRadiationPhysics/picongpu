/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/block/shared/st/Traits.hpp>

#    include <cstdint>
#    include <type_traits>

namespace alpaka
{
    //#############################################################################
    //! The GPU CUDA/HIP block shared memory allocator.
    class BlockSharedMemStUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStUniformCudaHipBuiltIn>
    {
    public:
        //-----------------------------------------------------------------------------
        //! Default constructor.
        BlockSharedMemStUniformCudaHipBuiltIn() = default;
        //-----------------------------------------------------------------------------
        //! Copy constructor.
        __device__ BlockSharedMemStUniformCudaHipBuiltIn(BlockSharedMemStUniformCudaHipBuiltIn const&) = delete;
        //-----------------------------------------------------------------------------
        //! Move constructor.
        __device__ BlockSharedMemStUniformCudaHipBuiltIn(BlockSharedMemStUniformCudaHipBuiltIn&&) = delete;
        //-----------------------------------------------------------------------------
        //! Copy assignment operator.
        __device__ auto operator=(BlockSharedMemStUniformCudaHipBuiltIn const&)
            -> BlockSharedMemStUniformCudaHipBuiltIn& = delete;
        //-----------------------------------------------------------------------------
        //! Move assignment operator.
        __device__ auto operator=(BlockSharedMemStUniformCudaHipBuiltIn&&)
            -> BlockSharedMemStUniformCudaHipBuiltIn& = delete;
        //-----------------------------------------------------------------------------
        //! Destructor.
        /*virtual*/ ~BlockSharedMemStUniformCudaHipBuiltIn() = default;
    };

    namespace traits
    {
        //#############################################################################
        template<typename T, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, BlockSharedMemStUniformCudaHipBuiltIn>
        {
            //-----------------------------------------------------------------------------
            __device__ static auto declareVar(BlockSharedMemStUniformCudaHipBuiltIn const&) -> T&
            {
                __shared__ uint8_t shMem alignas(alignof(T))[sizeof(T)];
                return *(reinterpret_cast<T*>(shMem));
            }
        };
        //#############################################################################
        template<>
        struct FreeSharedVars<BlockSharedMemStUniformCudaHipBuiltIn>
        {
            //-----------------------------------------------------------------------------
            __device__ static auto freeVars(BlockSharedMemStUniformCudaHipBuiltIn const&) -> void
            {
                // Nothing to do. CUDA/HIP block shared memory is automatically freed when all threads left the block.
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
