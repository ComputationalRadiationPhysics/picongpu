/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/block/shared/dyn/Traits.hpp>

#include <type_traits>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                //#############################################################################
                //! The GPU CUDA/HIP block shared memory allocator.
                class BlockSharedMemDynUniformCudaHipBuiltIn : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynUniformCudaHipBuiltIn>
                {
                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynUniformCudaHipBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    __device__ BlockSharedMemDynUniformCudaHipBuiltIn(BlockSharedMemDynUniformCudaHipBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    __device__ BlockSharedMemDynUniformCudaHipBuiltIn(BlockSharedMemDynUniformCudaHipBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    __device__ auto operator=(BlockSharedMemDynUniformCudaHipBuiltIn const &) -> BlockSharedMemDynUniformCudaHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    __device__ auto operator=(BlockSharedMemDynUniformCudaHipBuiltIn &&) -> BlockSharedMemDynUniformCudaHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynUniformCudaHipBuiltIn() = default;
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynUniformCudaHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        __device__ static auto getMem(
                            block::shared::dyn::BlockSharedMemDynUniformCudaHipBuiltIn const &)
                        -> T *
                        {
                            // Because unaligned access to variables is not allowed in device code,
                            // we have to use the widest possible type to have all types aligned correctly.
                            // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
                            // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types
                            extern __shared__ float4 shMem[];
                            return reinterpret_cast<T *>(shMem);
                        }
                    };
                }
            }
        }
    }
}

#endif
