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

#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/size/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Common.hpp>

#include <iosfwd>

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! A basic class holding the work division as grid block extent, block thread and thread element extent.
        template<
            typename TDim,
            typename TSize>
        class WorkDivMembers
        {
        public:
            using WorkDivBase = WorkDivMembers;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST_ACC WorkDivMembers() = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TGridBlockExtent,
                typename TBlockThreadExtent,
                typename TThreadElemExtent>
            ALPAKA_FN_HOST_ACC explicit WorkDivMembers(
                TGridBlockExtent const & gridBlockExtent = TGridBlockExtent(),
                TBlockThreadExtent const & blockThreadExtent = TBlockThreadExtent(),
                TThreadElemExtent const & threadElemExtent = TThreadElemExtent()) :
                m_gridBlockExtent(extent::getExtentVecEnd<TDim>(gridBlockExtent)),
                m_blockThreadExtent(extent::getExtentVecEnd<TDim>(blockThreadExtent)),
                m_threadElemExtent(extent::getExtentVecEnd<TDim>(threadElemExtent))
            {}
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC explicit WorkDivMembers(
                WorkDivMembers const & other) :
                    m_gridBlockExtent(other.m_gridBlockExtent),
                    m_blockThreadExtent(other.m_blockThreadExtent),
                    m_threadElemExtent(other.m_threadElemExtent)
            {}
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST_ACC explicit WorkDivMembers(
                TWorkDiv const & other) :
                    m_gridBlockExtent(vec::subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other))),
                    m_blockThreadExtent(vec::subVecEnd<TDim>(getWorkDiv<Block, Threads>(other))),
                    m_threadElemExtent(vec::subVecEnd<TDim>(getWorkDiv<Thread, Elems>(other)))
            {}
            //-----------------------------------------------------------------------------
            WorkDivMembers(WorkDivMembers &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(WorkDivMembers const &) -> WorkDivMembers & = default;
            //-----------------------------------------------------------------------------
            auto operator=(WorkDivMembers &&) -> WorkDivMembers & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST_ACC auto operator=(
                TWorkDiv const & other)
            -> WorkDivMembers<TDim, TSize> &
            {
                m_gridBlockExtent = vec::subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other));
                m_blockThreadExtent = vec::subVecEnd<TDim>(getWorkDiv<Block, Threads>(other));
                m_threadElemExtent = vec::subVecEnd<TDim>(getWorkDiv<Thread, Elems>(other));
                return *this;
            }
            //-----------------------------------------------------------------------------
            /*virtual*/ ~WorkDivMembers() = default;

        public:
            vec::Vec<TDim, TSize> m_gridBlockExtent;
            vec::Vec<TDim, TSize> m_blockThreadExtent;
            vec::Vec<TDim, TSize> m_threadElemExtent;
        };

        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST auto operator<<(
            std::ostream & os,
            WorkDivMembers<TDim, TSize> const & workDiv)
        -> std::ostream &
        {
            return (os
                << "{gridBlockExtent: " << workDiv.m_gridBlockExtent
                << ", blockThreadExtent: " << workDiv.m_blockThreadExtent
                << ", threadElemExtent: " << workDiv.m_threadElemExtent
                << "}");
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The WorkDivMembers dimension get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                workdiv::WorkDivMembers<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The WorkDivMembers size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                workdiv::WorkDivMembers<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
    namespace workdiv
    {
        namespace traits
        {
            //#############################################################################
            //! The WorkDivMembers grid block extent trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivMembers<TDim, TSize>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    WorkDivMembers<TDim, TSize> const & workDiv)
                -> vec::Vec<TDim, TSize>
                {
                    return workDiv.m_gridBlockExtent;
                }
            };

            //#############################################################################
            //! The WorkDivMembers block thread extent trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivMembers<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    WorkDivMembers<TDim, TSize> const & workDiv)
                -> vec::Vec<TDim, TSize>
                {
                    return workDiv.m_blockThreadExtent;
                }
            };

            //#############################################################################
            //! The WorkDivMembers thread element extent trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivMembers<TDim, TSize>,
                origin::Thread,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of elements in each dimension of a thread.
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    WorkDivMembers<TDim, TSize> const & workDiv)
                -> vec::Vec<TDim, TSize>
                {
                    return workDiv.m_threadElemExtent;
                }
            };
        }
    }
}
