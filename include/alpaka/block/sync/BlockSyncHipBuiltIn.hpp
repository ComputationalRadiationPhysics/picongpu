/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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
