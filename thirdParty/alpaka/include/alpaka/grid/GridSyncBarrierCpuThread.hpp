/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BarrierThread.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/grid/Traits.hpp"

#include <thread>

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

namespace alpaka
{
    //! The thread id map barrier grid synchronization.
    template<typename TIdx>
    class GridSyncBarrierThread : public interface::Implements<ConceptGridSync, GridSyncBarrierThread<TIdx>>
    {
    public:
        using Barrier = core::threads::BarrierThread<TIdx>;

        ALPAKA_FN_HOST GridSyncBarrierThread(TIdx const& blockThreadCount) : m_barrier(blockThreadCount)
        {
        }

        Barrier mutable m_barrier;
    };

    namespace trait
    {
        template<typename TIdx>
        struct SyncGridThreads<GridSyncBarrierThread<TIdx>>
        {
            ALPAKA_FN_HOST static auto syncGridThreads(GridSyncBarrierThread<TIdx> const& gridSync) -> void
            {
                gridSync.m_barrier.wait();
            }
        };

    } // namespace trait
} // namespace alpaka

#endif
