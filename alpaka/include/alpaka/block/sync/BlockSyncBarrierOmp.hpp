/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef _OPENMP

#include <alpaka/block/sync/Traits.hpp>

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The OpenMP barrier block synchronization.
            class BlockSyncBarrierOmp : public concepts::Implements<ConceptBlockSync, BlockSyncBarrierOmp>
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierOmp() :
                    m_generation(0u)
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierOmp(BlockSyncBarrierOmp const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierOmp(BlockSyncBarrierOmp &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierOmp const &) -> BlockSyncBarrierOmp & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierOmp &&) -> BlockSyncBarrierOmp & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~BlockSyncBarrierOmp() = default;

                std::uint8_t mutable m_generation;
                int mutable m_result[2];
            };

            namespace traits
            {
                //#############################################################################
                template<>
                struct SyncBlockThreads<
                    BlockSyncBarrierOmp>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto syncBlockThreads(
                        block::sync::BlockSyncBarrierOmp const & blockSync)
                    -> void
                    {
                        alpaka::ignore_unused(blockSync);

                        // NOTE: This waits for all threads in all blocks.
                        // If multiple blocks are executed in parallel this is not optimal.
                        #pragma omp barrier
                    }
                };

                namespace detail
                {
                    //#############################################################################
                    template<
                        typename TOp>
                    struct AtomicOp;
                    //#############################################################################
                    template<>
                    struct AtomicOp<
                        block::sync::op::Count>
                    {
                        void operator()(int& result, bool value)
                        {
                            #pragma omp atomic
                            result += static_cast<int>(value);
                        }
                    };
                    //#############################################################################
                    template<>
                    struct AtomicOp<
                        block::sync::op::LogicalAnd>
                    {
                        void operator()(int& result, bool value)
                        {
                            #pragma omp atomic
                            result &= static_cast<int>(value);
                        }
                    };
                    //#############################################################################
                    template<>
                    struct AtomicOp<
                        block::sync::op::LogicalOr>
                    {
                        void operator()(int& result, bool value)
                        {
                            #pragma omp atomic
                            result |= static_cast<int>(value);
                        }
                    };
                }

                //#############################################################################
                template<
                    typename TOp>
                struct SyncBlockThreadsPredicate<
                    TOp,
                    BlockSyncBarrierOmp>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncBarrierOmp const & blockSync,
                        int predicate)
                    -> int
                    {
                        // The first thread initializes the value.
                        // There is an implicit barrier at the end of omp single.
                        // NOTE: This code is executed only once for all OpenMP threads.
                        // If multiple blocks with multiple threads are executed in parallel
                        // this reduction is executed only for one block!
                        #pragma omp single
                        {
                            ++blockSync.m_generation;
                            blockSync.m_result[blockSync.m_generation % 2u] = TOp::InitialValue;
                        }

                        auto const generationMod2(blockSync.m_generation % 2u);
                        int& result(blockSync.m_result[generationMod2]);
                        bool const predicateBool(predicate != 0);

                        detail::AtomicOp<TOp>()(result, predicateBool);

                        // Wait for all threads to write their predicate into the vector.
                        // NOTE: This waits for all threads in all blocks.
                        // If multiple blocks are executed in parallel this is not optimal.
                        #pragma omp barrier

                        return blockSync.m_result[generationMod2];
                    }
                };
            }
        }
    }
}

#endif
