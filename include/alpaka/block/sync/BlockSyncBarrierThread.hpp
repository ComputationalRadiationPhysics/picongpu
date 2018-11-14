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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#include <alpaka/block/sync/Traits.hpp>

#include <alpaka/core/BarrierThread.hpp>

#include <alpaka/core/Common.hpp>

#include <thread>
#include <mutex>
#include <map>

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The thread id map barrier block synchronization.
            template<
                typename TIdx>
            class BlockSyncBarrierThread
            {
            public:
                using BlockSyncBase = BlockSyncBarrierThread;

                using Barrier = core::threads::BarrierThread<TIdx>;
                using BarrierWithPredicate = core::threads::BarrierThreadWithPredicate<TIdx>;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierThread(
                    TIdx const & blockThreadCount) :
                        m_barrier(blockThreadCount),
                        m_barrierWithPredicate(blockThreadCount)
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierThread(BlockSyncBarrierThread const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierThread(BlockSyncBarrierThread &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierThread const &) -> BlockSyncBarrierThread & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierThread &&) -> BlockSyncBarrierThread & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~BlockSyncBarrierThread() = default;

                Barrier mutable m_barrier;
                BarrierWithPredicate mutable m_barrierWithPredicate;
            };

            namespace traits
            {
                //#############################################################################
                template<
                    typename TIdx>
                struct SyncBlockThreads<
                    BlockSyncBarrierThread<TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto syncBlockThreads(
                        block::sync::BlockSyncBarrierThread<TIdx> const & blockSync)
                    -> void
                    {
                        blockSync.m_barrier.wait();
                    }
                };

                //#############################################################################
                template<
                    typename TOp,
                    typename TIdx>
                struct SyncBlockThreadsPredicate<
                    TOp,
                    BlockSyncBarrierThread<TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncBarrierThread<TIdx> const & blockSync,
                        int predicate)
                    -> int
                    {
                        return blockSync.m_barrierWithPredicate.template wait<TOp>(predicate);
                    }
                };
            }
        }
    }
}

#endif
