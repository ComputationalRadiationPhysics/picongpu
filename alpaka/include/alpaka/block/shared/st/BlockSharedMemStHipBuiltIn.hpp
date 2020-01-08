/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

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
