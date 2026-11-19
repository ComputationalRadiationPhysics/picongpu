/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BarrierTbb.h"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/grid/Traits.hpp"

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

namespace alpaka
{
    //! The thread id map barrier grid synchronization for TBB.
    template<typename TIdx>
    class GridSyncBarrierTbb : public interface::Implements<ConceptGridSync, GridSyncBarrierTbb<TIdx>>
    {
    public:
        using Barrier = core::tbb::BarrierThread<TIdx>;

        // Get reference to the barrier from the outside because we need it to be shared between blocks
        ALPAKA_FN_HOST explicit GridSyncBarrierTbb(Barrier& barrier)
        {
            m_barrier = &barrier;
        }

        Barrier* m_barrier;
    };

    namespace trait
    {
        template<typename TIdx>
        struct SyncGridThreads<GridSyncBarrierTbb<TIdx>>
        {
            ALPAKA_FN_HOST static auto syncGridThreads(GridSyncBarrierTbb<TIdx> const& gridSync) -> void
            {
                gridSync.m_barrier->wait();
            }
        };

    } // namespace trait
} // namespace alpaka

#endif
