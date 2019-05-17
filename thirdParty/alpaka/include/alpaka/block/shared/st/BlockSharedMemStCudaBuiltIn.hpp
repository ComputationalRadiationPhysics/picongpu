/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/block/shared/st/Traits.hpp>

#include <type_traits>
#include <cstdint>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace st
            {
                //#############################################################################
                //! The GPU CUDA block shared memory allocator.
                class BlockSharedMemStCudaBuiltIn
                {
                public:
                    using BlockSharedMemStBase = BlockSharedMemStCudaBuiltIn;

                    //-----------------------------------------------------------------------------
                    BlockSharedMemStCudaBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    __device__ BlockSharedMemStCudaBuiltIn(BlockSharedMemStCudaBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    __device__ BlockSharedMemStCudaBuiltIn(BlockSharedMemStCudaBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    __device__ auto operator=(BlockSharedMemStCudaBuiltIn const &) -> BlockSharedMemStCudaBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    __device__ auto operator=(BlockSharedMemStCudaBuiltIn &&) -> BlockSharedMemStCudaBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemStCudaBuiltIn() = default;
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStCudaBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        __device__ static auto allocVar(
                            block::shared::st::BlockSharedMemStCudaBuiltIn const &)
                        -> T &
                        {
                            __shared__ uint8_t shMem alignas(alignof(T)) [sizeof(T)];
                            return *(
                                reinterpret_cast<T*>( shMem ));
                        }
                    };
                    //#############################################################################
                    template<>
                    struct FreeMem<
                        BlockSharedMemStCudaBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        __device__ static auto freeMem(
                            block::shared::st::BlockSharedMemStCudaBuiltIn const &)
                        -> void
                        {
                            // Nothing to do. CUDA block shared memory is automatically freed when all threads left the block.
                        }
                    };
                }
            }
        }
    }
}

#endif
