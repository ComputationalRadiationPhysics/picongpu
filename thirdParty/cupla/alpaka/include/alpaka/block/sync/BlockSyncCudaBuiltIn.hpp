/* Copyright 2019 Benjamin Worpitz
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

#include <alpaka/block/sync/Traits.hpp>

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The GPU CUDA block synchronization.
            class BlockSyncCudaBuiltIn : public concepts::Implements<ConceptBlockSync, BlockSyncCudaBuiltIn>
            {
            public:
                //-----------------------------------------------------------------------------
                BlockSyncCudaBuiltIn() = default;
                //-----------------------------------------------------------------------------
                __device__ BlockSyncCudaBuiltIn(BlockSyncCudaBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                __device__ BlockSyncCudaBuiltIn(BlockSyncCudaBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(BlockSyncCudaBuiltIn const &) -> BlockSyncCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(BlockSyncCudaBuiltIn &&) -> BlockSyncCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~BlockSyncCudaBuiltIn() = default;
            };

            namespace traits
            {
                //#############################################################################
                template<>
                struct SyncBlockThreads<
                    BlockSyncCudaBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto syncBlockThreads(
                        block::sync::BlockSyncCudaBuiltIn const & /*blockSync*/)
                    -> void
                    {
                        __syncthreads();
                    }
                };

                //#############################################################################
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::Count,
                    BlockSyncCudaBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncCudaBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
                        return __syncthreads_count(predicate);
                    }
                };

                //#############################################################################
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalAnd,
                    BlockSyncCudaBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncCudaBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
                        return __syncthreads_and(predicate);
                    }
                };

                //#############################################################################
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalOr,
                    BlockSyncCudaBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncCudaBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
                        return __syncthreads_or(predicate);
                    }
                };
            }
        }
    }
}

#endif
