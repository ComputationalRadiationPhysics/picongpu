/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#    include <alpaka/block/sync/Traits.hpp>
#    include <alpaka/core/BarrierThread.hpp>
#    include <alpaka/core/Common.hpp>

#    include <map>
#    include <mutex>
#    include <thread>

namespace alpaka
{
    //#############################################################################
    //! The thread id map barrier block synchronization.
    template<typename TIdx>
    class BlockSyncBarrierThread : public concepts::Implements<ConceptBlockSync, BlockSyncBarrierThread<TIdx>>
    {
    public:
        using Barrier = core::threads::BarrierThread<TIdx>;
        using BarrierWithPredicate = core::threads::BarrierThreadWithPredicate<TIdx>;

        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST BlockSyncBarrierThread(TIdx const& blockThreadCount)
            : m_barrier(blockThreadCount)
            , m_barrierWithPredicate(blockThreadCount)
        {
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST BlockSyncBarrierThread(BlockSyncBarrierThread const&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST BlockSyncBarrierThread(BlockSyncBarrierThread&&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(BlockSyncBarrierThread const&) -> BlockSyncBarrierThread& = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(BlockSyncBarrierThread&&) -> BlockSyncBarrierThread& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSyncBarrierThread() = default;

        Barrier mutable m_barrier;
        BarrierWithPredicate mutable m_barrierWithPredicate;
    };

    namespace traits
    {
        //#############################################################################
        template<typename TIdx>
        struct SyncBlockThreads<BlockSyncBarrierThread<TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto syncBlockThreads(BlockSyncBarrierThread<TIdx> const& blockSync) -> void
            {
                blockSync.m_barrier.wait();
            }
        };

        //#############################################################################
        template<typename TOp, typename TIdx>
        struct SyncBlockThreadsPredicate<TOp, BlockSyncBarrierThread<TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                BlockSyncBarrierThread<TIdx> const& blockSync,
                int predicate) -> int
            {
                return blockSync.m_barrierWithPredicate.template wait<TOp>(predicate);
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
