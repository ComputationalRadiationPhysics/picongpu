/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
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
                //! The GPU HIP block shared memory allocator.
                class BlockSharedMemDynHipBuiltIn
                {
                public:
                    using BlockSharedMemDynBase = BlockSharedMemDynHipBuiltIn;

                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    ALPAKA_FN_HOST_ACC BlockSharedMemDynHipBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    __device__ BlockSharedMemDynHipBuiltIn(BlockSharedMemDynHipBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    __device__ BlockSharedMemDynHipBuiltIn(BlockSharedMemDynHipBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    __device__ auto operator=(BlockSharedMemDynHipBuiltIn const &) -> BlockSharedMemDynHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    __device__ auto operator=(BlockSharedMemDynHipBuiltIn &&) -> BlockSharedMemDynHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    /*virtual*/ ALPAKA_FN_HOST_ACC ~BlockSharedMemDynHipBuiltIn() = default;
                };

                namespace traits
                {
                    //#############################################################################
                    //!
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------

                        __device__ static auto getMem(
                            block::shared::dyn::BlockSharedMemDynHipBuiltIn const &)
                        -> T *
                        {
                            // Because unaligned access to variables is not allowed in device code,
                            // we have to use the widest possible type to have all types aligned correctly.
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
