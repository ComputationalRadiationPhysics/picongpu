/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/block/sync/Traits.hpp>

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The GPU HIP block synchronization.
            class BlockSyncHipBuiltIn
            {
            public:
                using BlockSyncBase = BlockSyncHipBuiltIn;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                ALPAKA_FN_HOST_ACC BlockSyncHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                __device__ BlockSyncHipBuiltIn(BlockSyncHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                __device__ BlockSyncHipBuiltIn(BlockSyncHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                __device__ auto operator=(BlockSyncHipBuiltIn const &) -> BlockSyncHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                __device__ auto operator=(BlockSyncHipBuiltIn &&) -> BlockSyncHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                /*virtual*/ ALPAKA_FN_HOST_ACC ~BlockSyncHipBuiltIn() = default;
            };

            namespace traits
            {
                //#############################################################################
                //!
                template<>
                struct SyncBlockThreads<
                    BlockSyncHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------

                    __device__ static auto syncBlockThreads(
                        block::sync::BlockSyncHipBuiltIn const & /*blockSync*/)
                    -> void
                    {
                        __syncthreads();
                    }
                };

                //#############################################################################
                //!
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::Count,
                    BlockSyncHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------

                    __device__ static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncHipBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
#if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__==0 && BOOST_COMP_HCC
                        // workaround for unsupported syncthreads_* operation on HIP(HCC)
                        __shared__ int tmp;
                        __syncthreads();
                        if(threadIdx.x==0)
                            tmp=0;
                        __syncthreads();
                        if(predicate)
                            atomicAdd(&tmp, 1);
                        __syncthreads();

                        return tmp;
#else
                        return __syncthreads_count(predicate);
#endif
                    }
                };

                //#############################################################################
                //!
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalAnd,
                    BlockSyncHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------

                    __device__ static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncHipBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
#if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__==0 && BOOST_COMP_HCC
                        // workaround for unsupported syncthreads_* operation on HIP(HCC)
                        __shared__ int tmp;
                        __syncthreads();
                        if(threadIdx.x==0)
                            tmp=1;
                        __syncthreads();
                        if(!predicate)
                            atomicAnd(&tmp, 0);
                        __syncthreads();

                        return tmp;
#else
                        return __syncthreads_and(predicate);
#endif
                    }
                };

                //#############################################################################
                //!
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalOr,
                    BlockSyncHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------

                    __device__ static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncHipBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
#if defined(__HIP_ARCH_HAS_SYNC_THREAD_EXT__) && __HIP_ARCH_HAS_SYNC_THREAD_EXT__==0 && BOOST_COMP_HCC
                        // workaround for unsupported syncthreads_* operation on HIP(HCC)
                        __shared__ int tmp;
                        __syncthreads();
                        if(threadIdx.x==0)
                            tmp=0;
                        __syncthreads();
                        if(predicate)
                            atomicOr(&tmp, 1);
                        __syncthreads();

                        return tmp;
#else
                        return __syncthreads_or(predicate);
#endif
                    }
                };
            }
        }
    }
}

#endif
