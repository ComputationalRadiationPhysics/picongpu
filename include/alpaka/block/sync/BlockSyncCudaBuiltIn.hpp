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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

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
            class BlockSyncCudaBuiltIn
            {
            public:
                using BlockSyncBase = BlockSyncCudaBuiltIn;

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
