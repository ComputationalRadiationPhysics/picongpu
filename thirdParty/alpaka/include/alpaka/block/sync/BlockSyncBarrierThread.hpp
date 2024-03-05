/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/sync/Traits.hpp"
#include "alpaka/core/BarrierThread.hpp"
#include "alpaka/core/Common.hpp"

#include <map>
#include <mutex>
#include <thread>

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

namespace alpaka
{
    //! The thread id map barrier block synchronization.
    template<typename TIdx>
    class BlockSyncBarrierThread : public concepts::Implements<ConceptBlockSync, BlockSyncBarrierThread<TIdx>>
    {
    public:
        using Barrier = core::threads::BarrierThread<TIdx>;
        using BarrierWithPredicate = core::threads::BarrierThreadWithPredicate<TIdx>;

        ALPAKA_FN_HOST BlockSyncBarrierThread(TIdx const& blockThreadCount)
            : m_barrier(blockThreadCount)
            , m_barrierWithPredicate(blockThreadCount)
        {
        }

        Barrier mutable m_barrier;
        BarrierWithPredicate mutable m_barrierWithPredicate;
    };

    namespace trait
    {
        template<typename TIdx>
        struct SyncBlockThreads<BlockSyncBarrierThread<TIdx>>
        {
            ALPAKA_FN_HOST static auto syncBlockThreads(BlockSyncBarrierThread<TIdx> const& blockSync) -> void
            {
                blockSync.m_barrier.wait();
            }
        };

        template<typename TOp, typename TIdx>
        struct SyncBlockThreadsPredicate<TOp, BlockSyncBarrierThread<TIdx>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                BlockSyncBarrierThread<TIdx> const& blockSync,
                int predicate) -> int
            {
                return blockSync.m_barrierWithPredicate.template wait<TOp>(predicate);
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
