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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED

#include <alpaka/idx/Traits.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Fibers.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

#include <map>

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The fibers accelerator index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxBtRefFiberIdMap
            {
            public:
                using IdxBtBase = IdxBtRefFiberIdMap;

                using FiberIdToIdxMap = std::map<boost::fibers::fiber::id, vec::Vec<TDim, TIdx>>;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST IdxBtRefFiberIdMap(
                    FiberIdToIdxMap const & mFibersToIndices) :
                    m_fibersToIndices(mFibersToIndices)
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST IdxBtRefFiberIdMap(IdxBtRefFiberIdMap const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST IdxBtRefFiberIdMap(IdxBtRefFiberIdMap &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(IdxBtRefFiberIdMap const &) -> IdxBtRefFiberIdMap & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(IdxBtRefFiberIdMap &&) -> IdxBtRefFiberIdMap & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxBtRefFiberIdMap() = default;

            public:
                FiberIdToIdxMap const & m_fibersToIndices; //!< The mapping of fiber id's to fiber indices.
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::bt::IdxBtRefFiberIdMap<TDim, TIdx>>
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
            //! The CPU fibers accelerator block thread index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::bt::IdxBtRefFiberIdMap<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST static auto getIdx(
                    idx::bt::IdxBtRefFiberIdMap<TDim, TIdx> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(workDiv);
                    auto const fiberId(boost::this_fiber::get_id());
                    auto const fiberEntry(idx.m_fibersToIndices.find(fiberId));
                    ALPAKA_ASSERT(fiberEntry != idx.m_fibersToIndices.end());
                    return fiberEntry->second;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator block thread index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtRefFiberIdMap<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
