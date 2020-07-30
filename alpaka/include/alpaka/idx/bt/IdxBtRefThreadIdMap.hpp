/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#include <alpaka/idx/Traits.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

#include <thread>
#include <map>

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The threads accelerator index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxBtRefThreadIdMap : public concepts::Implements<ConceptIdxBt, IdxBtRefThreadIdMap<TDim, TIdx>>
            {
            public:
                using ThreadIdToIdxMap = std::map<std::thread::id, vec::Vec<TDim, TIdx>>;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST IdxBtRefThreadIdMap(
                    ThreadIdToIdxMap const & mThreadToIndices) :
                    m_threadToIndexMap(mThreadToIndices)
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST IdxBtRefThreadIdMap(IdxBtRefThreadIdMap const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST IdxBtRefThreadIdMap(IdxBtRefThreadIdMap &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(IdxBtRefThreadIdMap const &) -> IdxBtRefThreadIdMap & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(IdxBtRefThreadIdMap &&) -> IdxBtRefThreadIdMap & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxBtRefThreadIdMap() = default;

            public:
                ThreadIdToIdxMap const & m_threadToIndexMap;   //!< The mapping of thread id's to thread indices.
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::bt::IdxBtRefThreadIdMap<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator block thread index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::bt::IdxBtRefThreadIdMap<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST static auto getIdx(
                    idx::bt::IdxBtRefThreadIdMap<TDim, TIdx> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(workDiv);
                    auto const threadId(std::this_thread::get_id());
                    auto const threadEntry(idx.m_threadToIndexMap.find(threadId));
                    ALPAKA_ASSERT(threadEntry != idx.m_threadToIndexMap.end());
                    return threadEntry->second;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator block thread index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtRefThreadIdMap<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
