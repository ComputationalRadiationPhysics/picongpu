/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
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
                //! The GPU CUDA block shared memory allocator.
                class BlockSharedMemDynCudaBuiltIn : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynCudaBuiltIn>
                {
                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynCudaBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    __device__ BlockSharedMemDynCudaBuiltIn(BlockSharedMemDynCudaBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    __device__ BlockSharedMemDynCudaBuiltIn(BlockSharedMemDynCudaBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    __device__ auto operator=(BlockSharedMemDynCudaBuiltIn const &) -> BlockSharedMemDynCudaBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    __device__ auto operator=(BlockSharedMemDynCudaBuiltIn &&) -> BlockSharedMemDynCudaBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynCudaBuiltIn() = default;
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynCudaBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        __device__ static auto getMem(
                            block::shared::dyn::BlockSharedMemDynCudaBuiltIn const &)
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
