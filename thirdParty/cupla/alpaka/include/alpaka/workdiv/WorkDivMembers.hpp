/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/Traits.hpp>

#include <iosfwd>

namespace alpaka
{
    //#############################################################################
    //! A basic class holding the work division as grid block extent, block thread and thread element extent.
    template<typename TDim, typename TIdx>
    class WorkDivMembers : public concepts::Implements<ConceptWorkDiv, WorkDivMembers<TDim, TIdx>>
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST_ACC WorkDivMembers() = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TGridBlockExtent, typename TBlockThreadExtent, typename TThreadElemExtent>
        ALPAKA_FN_HOST_ACC explicit WorkDivMembers(
            TGridBlockExtent const& gridBlockExtent = TGridBlockExtent(),
            TBlockThreadExtent const& blockThreadExtent = TBlockThreadExtent(),
            TThreadElemExtent const& threadElemExtent = TThreadElemExtent())
            : m_gridBlockExtent(extent::getExtentVecEnd<TDim>(gridBlockExtent))
            , m_blockThreadExtent(extent::getExtentVecEnd<TDim>(blockThreadExtent))
            , m_threadElemExtent(extent::getExtentVecEnd<TDim>(threadElemExtent))
        {
        }
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC explicit WorkDivMembers(WorkDivMembers const& other)
            : m_gridBlockExtent(other.m_gridBlockExtent)
            , m_blockThreadExtent(other.m_blockThreadExtent)
            , m_threadElemExtent(other.m_threadElemExtent)
        {
        }
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TWorkDiv>
        ALPAKA_FN_HOST_ACC explicit WorkDivMembers(TWorkDiv const& other)
            : m_gridBlockExtent(subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other)))
            , m_blockThreadExtent(subVecEnd<TDim>(getWorkDiv<Block, Threads>(other)))
            , m_threadElemExtent(subVecEnd<TDim>(getWorkDiv<Thread, Elems>(other)))
        {
        }
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC
        WorkDivMembers(WorkDivMembers&&) = default;
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC
        auto operator=(WorkDivMembers const&) -> WorkDivMembers& = default;
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC
        auto operator=(WorkDivMembers&&) -> WorkDivMembers& = default;
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TWorkDiv>
        ALPAKA_FN_HOST_ACC auto operator=(TWorkDiv const& other) -> WorkDivMembers<TDim, TIdx>&
        {
            m_gridBlockExtent = subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other));
            m_blockThreadExtent = subVecEnd<TDim>(getWorkDiv<Block, Threads>(other));
            m_threadElemExtent = subVecEnd<TDim>(getWorkDiv<Thread, Elems>(other));
            return *this;
        }
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        /*virtual*/ ALPAKA_FN_HOST_ACC ~WorkDivMembers() = default;

    public:
        Vec<TDim, TIdx> m_gridBlockExtent;
        Vec<TDim, TIdx> m_blockThreadExtent;
        Vec<TDim, TIdx> m_threadElemExtent;
    };

    //-----------------------------------------------------------------------------
    template<typename TDim, typename TIdx>
    ALPAKA_FN_HOST auto operator<<(std::ostream& os, WorkDivMembers<TDim, TIdx> const& workDiv) -> std::ostream&
    {
        return (
            os << "{gridBlockExtent: " << workDiv.m_gridBlockExtent << ", blockThreadExtent: "
               << workDiv.m_blockThreadExtent << ", threadElemExtent: " << workDiv.m_threadElemExtent << "}");
    }

    namespace traits
    {
        //#############################################################################
        //! The WorkDivMembers dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<WorkDivMembers<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The WorkDivMembers idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<WorkDivMembers<TDim, TIdx>>
        {
            using type = TIdx;
        };

        //#############################################################################
        //! The WorkDivMembers grid block extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivMembers<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of blocks in each dimension of the grid.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(WorkDivMembers<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
            {
                return workDiv.m_gridBlockExtent;
            }
        };

        //#############################################################################
        //! The WorkDivMembers block thread extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivMembers<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of threads in each dimension of a block.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(WorkDivMembers<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
            {
                return workDiv.m_blockThreadExtent;
            }
        };

        //#############################################################################
        //! The WorkDivMembers thread element extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivMembers<TDim, TIdx>, origin::Thread, unit::Elems>
        {
            //-----------------------------------------------------------------------------
            //! \return The number of elements in each dimension of a thread.
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC static auto getWorkDiv(WorkDivMembers<TDim, TIdx> const& workDiv) -> Vec<TDim, TIdx>
            {
                return workDiv.m_threadElemExtent;
            }
        };
    } // namespace traits
} // namespace alpaka
