/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Fibers.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <map>

namespace alpaka
{
    namespace bt
    {
        //#############################################################################
        //! The fibers accelerator index provider.
        template<typename TDim, typename TIdx>
        class IdxBtRefFiberIdMap : public concepts::Implements<ConceptIdxBt, IdxBtRefFiberIdMap<TDim, TIdx>>
        {
        public:
            using FiberIdToIdxMap = std::map<boost::fibers::fiber::id, Vec<TDim, TIdx>>;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST IdxBtRefFiberIdMap(FiberIdToIdxMap const& mFibersToIndices)
                : m_fibersToIndices(mFibersToIndices)
            {
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST IdxBtRefFiberIdMap(IdxBtRefFiberIdMap const&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST IdxBtRefFiberIdMap(IdxBtRefFiberIdMap&&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(IdxBtRefFiberIdMap const&) -> IdxBtRefFiberIdMap& = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(IdxBtRefFiberIdMap&&) -> IdxBtRefFiberIdMap& = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~IdxBtRefFiberIdMap() = default;

        public:
            FiberIdToIdxMap const& m_fibersToIndices; //!< The mapping of fiber id's to fiber indices.
        };
    } // namespace bt

    namespace traits
    {
        //#############################################################################
        //! The CPU fibers accelerator index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<bt::IdxBtRefFiberIdMap<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The CPU fibers accelerator block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtRefFiberIdMap<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            ALPAKA_FN_HOST static auto getIdx(bt::IdxBtRefFiberIdMap<TDim, TIdx> const& idx, TWorkDiv const& workDiv)
                -> Vec<TDim, TIdx>
            {
                alpaka::ignore_unused(workDiv);
                auto const fiberId(boost::this_fiber::get_id());
                auto const fiberEntry(idx.m_fibersToIndices.find(fiberId));
                ALPAKA_ASSERT(fiberEntry != idx.m_fibersToIndices.end());
                return fiberEntry->second;
            }
        };

        //#############################################################################
        //! The CPU fibers accelerator block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtRefFiberIdMap<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#endif
