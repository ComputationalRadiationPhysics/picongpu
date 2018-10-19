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
                    ALPAKA_FN_ACC_CUDA_ONLY BlockSharedMemStCudaBuiltIn(BlockSharedMemStCudaBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY BlockSharedMemStCudaBuiltIn(BlockSharedMemStCudaBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY auto operator=(BlockSharedMemStCudaBuiltIn const &) -> BlockSharedMemStCudaBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY auto operator=(BlockSharedMemStCudaBuiltIn &&) -> BlockSharedMemStCudaBuiltIn & = delete;
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
                        ALPAKA_FN_ACC_CUDA_ONLY static auto allocVar(
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
                        ALPAKA_FN_ACC_CUDA_ONLY static auto freeMem(
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
