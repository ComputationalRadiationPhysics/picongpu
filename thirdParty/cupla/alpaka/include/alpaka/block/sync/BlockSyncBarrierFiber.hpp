/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

#    include <alpaka/block/sync/Traits.hpp>
#    include <alpaka/core/Common.hpp>
#    include <alpaka/core/Fibers.hpp>

#    include <map>
#    include <mutex>

namespace alpaka
{
    //#############################################################################
    //! The thread id map barrier block synchronization.
    template<typename TIdx>
    class BlockSyncBarrierFiber : public concepts::Implements<ConceptBlockSync, BlockSyncBarrierFiber<TIdx>>
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST BlockSyncBarrierFiber(TIdx const& blockThreadCount)
            : m_barrier(static_cast<std::size_t>(blockThreadCount))
            , m_threadCount(blockThreadCount)
            , m_curThreadCount(static_cast<TIdx>(0u))
            , m_generation(static_cast<TIdx>(0u))
        {
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST BlockSyncBarrierFiber(BlockSyncBarrierFiber const&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST BlockSyncBarrierFiber(BlockSyncBarrierFiber&&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(BlockSyncBarrierFiber const&) -> BlockSyncBarrierFiber& = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(BlockSyncBarrierFiber&&) -> BlockSyncBarrierFiber& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSyncBarrierFiber() = default;

        boost::fibers::barrier mutable m_barrier;

        TIdx mutable m_threadCount;
        TIdx mutable m_curThreadCount;
        TIdx mutable m_generation;
        int mutable m_result[2u];
    };

    namespace traits
    {
        //#############################################################################
        template<typename TIdx>
        struct SyncBlockThreads<BlockSyncBarrierFiber<TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto syncBlockThreads(BlockSyncBarrierFiber<TIdx> const& blockSync) -> void
            {
                blockSync.m_barrier.wait();
            }
        };

        //#############################################################################
        template<typename TOp, typename TIdx>
        struct SyncBlockThreadsPredicate<TOp, BlockSyncBarrierFiber<TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                BlockSyncBarrierFiber<TIdx> const& blockSync,
                int predicate) -> int
            {
                if(blockSync.m_curThreadCount == blockSync.m_threadCount)
                {
                    blockSync.m_curThreadCount = static_cast<TIdx>(0u);
                    ++blockSync.m_generation;
                }

                auto const generationMod2(blockSync.m_generation % static_cast<TIdx>(2u));

                // The first fiber will reset the value to the initial value.
                if(blockSync.m_curThreadCount == static_cast<TIdx>(0u))
                {
                    blockSync.m_result[generationMod2] = TOp::InitialValue;
                }

                ++blockSync.m_curThreadCount;

                // We do not have to lock because there is only ever one fiber active per block.
                blockSync.m_result[generationMod2] = TOp()(blockSync.m_result[generationMod2], predicate);

                // After all block threads have combined their values ...
                blockSync.m_barrier.wait();

                // ... the result can be returned.
                return blockSync.m_result[generationMod2];
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
