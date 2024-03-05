/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/sync/Traits.hpp"
#include "alpaka/core/Common.hpp"

#include <cstdint>

#ifdef _OPENMP

namespace alpaka
{
    //! The OpenMP barrier block synchronization.
    class BlockSyncBarrierOmp : public concepts::Implements<ConceptBlockSync, BlockSyncBarrierOmp>
    {
    public:
        std::uint8_t mutable m_generation = 0u;
        int mutable m_result[2];
    };

    namespace trait
    {
        template<>
        struct SyncBlockThreads<BlockSyncBarrierOmp>
        {
            ALPAKA_FN_HOST static auto syncBlockThreads(BlockSyncBarrierOmp const& /* blockSync */) -> void
            {
// NOTE: This waits for all threads in all blocks.
// If multiple blocks are executed in parallel this is not optimal.
#    pragma omp barrier
            }
        };

        namespace detail
        {
            template<typename TOp>
            struct AtomicOp;

            template<>
            struct AtomicOp<BlockCount>
            {
                void operator()(int& result, bool value)
                {
#    pragma omp atomic
                    result += static_cast<int>(value);
                }
            };

            template<>
            struct AtomicOp<BlockAnd>
            {
                void operator()(int& result, bool value)
                {
#    pragma omp atomic
                    result &= static_cast<int>(value);
                }
            };

            template<>
            struct AtomicOp<BlockOr>
            {
                void operator()(int& result, bool value)
                {
#    pragma omp atomic
                    result |= static_cast<int>(value);
                }
            };
        } // namespace detail

        template<typename TOp>
        struct SyncBlockThreadsPredicate<TOp, BlockSyncBarrierOmp>
        {
            ALPAKA_NO_HOST_ACC_WARNING

            ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(BlockSyncBarrierOmp const& blockSync, int predicate)
                -> int
            {
// The first thread initializes the value.
// There is an implicit barrier at the end of omp single.
// NOTE: This code is executed only once for all OpenMP threads.
// If multiple blocks with multiple threads are executed in parallel
// this reduction is executed only for one block!
#    pragma omp single
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
#    pragma omp barrier

                return blockSync.m_result[generationMod2];
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
