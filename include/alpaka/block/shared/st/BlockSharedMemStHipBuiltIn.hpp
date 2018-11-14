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

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
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
                //! The GPU HIP block shared memory allocator.
                class BlockSharedMemStHipBuiltIn
                {
                public:
                    using BlockSharedMemStBase = BlockSharedMemStHipBuiltIn;

                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    ALPAKA_FN_HOST_ACC BlockSharedMemStHipBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    __device__ BlockSharedMemStHipBuiltIn(BlockSharedMemStHipBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    __device__ BlockSharedMemStHipBuiltIn(BlockSharedMemStHipBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    __device__ auto operator=(BlockSharedMemStHipBuiltIn const &) -> BlockSharedMemStHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    __device__ auto operator=(BlockSharedMemStHipBuiltIn &&) -> BlockSharedMemStHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    /*virtual*/ ALPAKA_FN_HOST_ACC ~BlockSharedMemStHipBuiltIn() = default;
                };

                namespace traits
                {
                    //#############################################################################
                    //!
                    template<
                        typename T,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------

                        __device__ static auto allocVar(
                            block::shared::st::BlockSharedMemStHipBuiltIn const &)
                        -> T &
                        {
                            __shared__ uint8_t shMem alignas(alignof(T)) [sizeof(T)];
                            return *(
                                reinterpret_cast<T*>( shMem ));
                        }
                    };
                    //#############################################################################
                    //!
                    template<>
                    struct FreeMem<
                        BlockSharedMemStHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------

                        __device__ static auto freeMem(
                            block::shared::st::BlockSharedMemStHipBuiltIn const &)
                        -> void
                        {
                            // Nothing to do. HIP block shared memory is automatically freed when all threads left the block.
                        }
                    };
                }
            }
        }
    }
}

#endif
