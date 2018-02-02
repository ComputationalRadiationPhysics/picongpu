/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz, Rene Widera
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
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

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
                class BlockSharedMemDynCudaBuiltIn
                {
                public:
                    using BlockSharedMemDynBase = BlockSharedMemDynCudaBuiltIn;

                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynCudaBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY BlockSharedMemDynCudaBuiltIn(BlockSharedMemDynCudaBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY BlockSharedMemDynCudaBuiltIn(BlockSharedMemDynCudaBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY auto operator=(BlockSharedMemDynCudaBuiltIn const &) -> BlockSharedMemDynCudaBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY auto operator=(BlockSharedMemDynCudaBuiltIn &&) -> BlockSharedMemDynCudaBuiltIn & = delete;
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
                        ALPAKA_FN_ACC_CUDA_ONLY static auto getMem(
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
