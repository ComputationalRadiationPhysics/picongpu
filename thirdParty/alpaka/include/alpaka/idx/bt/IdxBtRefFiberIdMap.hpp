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

#include <alpaka/core/Fibers.hpp>

#include <boost/core/ignore_unused.hpp>

#include <map>
#include <cassert>

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
                typename TSize>
            class IdxBtRefFiberIdMap
            {
            public:
                using IdxBtBase = IdxBtRefFiberIdMap;

                using FiberIdToIdxMap = std::map<boost::fibers::fiber::id, vec::Vec<TDim, TSize>>;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtRefFiberIdMap(
                    FiberIdToIdxMap const & mFibersToIndices) :
                    m_fibersToIndices(mFibersToIndices)
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtRefFiberIdMap(IdxBtRefFiberIdMap const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtRefFiberIdMap(IdxBtRefFiberIdMap &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtRefFiberIdMap const &) -> IdxBtRefFiberIdMap & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtRefFiberIdMap &&) -> IdxBtRefFiberIdMap & = delete;
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
                typename TSize>
            struct DimType<
                idx::bt::IdxBtRefFiberIdMap<TDim, TSize>>
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
                typename TSize>
            struct GetIdx<
                idx::bt::IdxBtRefFiberIdMap<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_NO_CUDA static auto getIdx(
                    idx::bt::IdxBtRefFiberIdMap<TDim, TSize> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TSize>
                {
                    boost::ignore_unused(workDiv);
                    auto const fiberId(boost::this_fiber::get_id());
                    auto const fiberEntry(idx.m_fibersToIndices.find(fiberId));
                    assert(fiberEntry != idx.m_fibersToIndices.end());
                    return fiberEntry->second;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator block thread index size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                idx::bt::IdxBtRefFiberIdMap<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
