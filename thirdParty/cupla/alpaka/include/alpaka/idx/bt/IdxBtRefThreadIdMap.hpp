/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <map>
#include <thread>

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

namespace alpaka
{
    namespace bt
    {
        //! The threads accelerator index provider.
        template<typename TDim, typename TIdx>
        class IdxBtRefThreadIdMap : public concepts::Implements<ConceptIdxBt, IdxBtRefThreadIdMap<TDim, TIdx>>
        {
        public:
            using ThreadIdToIdxMap = std::map<std::thread::id, Vec<TDim, TIdx>>;

            ALPAKA_FN_HOST IdxBtRefThreadIdMap(ThreadIdToIdxMap const& mThreadToIndices)
                : m_threadToIndexMap(mThreadToIndices)
            {
            }

            ALPAKA_FN_HOST IdxBtRefThreadIdMap(IdxBtRefThreadIdMap const&) = delete;
            ALPAKA_FN_HOST auto operator=(IdxBtRefThreadIdMap const&) -> IdxBtRefThreadIdMap& = delete;

        public:
            ThreadIdToIdxMap const& m_threadToIndexMap; //!< The mapping of thread id's to thread indices.
        };
    } // namespace bt

    namespace trait
    {
        //! The CPU threads accelerator index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<bt::IdxBtRefThreadIdMap<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The CPU threads accelerator block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtRefThreadIdMap<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            ALPAKA_FN_HOST static auto getIdx(
                bt::IdxBtRefThreadIdMap<TDim, TIdx> const& idx,
                TWorkDiv const& /* workDiv */) -> Vec<TDim, TIdx>
            {
                auto const threadId = std::this_thread::get_id();
                auto const threadEntry = idx.m_threadToIndexMap.find(threadId);
                ALPAKA_ASSERT(threadEntry != std::end(idx.m_threadToIndexMap));
                return threadEntry->second;
            }
        };

        //! The CPU threads accelerator block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtRefThreadIdMap<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka

#endif
