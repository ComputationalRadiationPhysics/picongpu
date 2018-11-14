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

#include <alpaka/meta/IsStrictBase.hpp>

#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Common.hpp>

#include <alpaka/vec/Vec.hpp>

#include <alpaka/idx/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/workdiv/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <boost/config.hpp>

#include <utility>

namespace alpaka
{
    namespace idx
    {
        //-----------------------------------------------------------------------------
        //! Get the indices requested.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOrigin,
            typename TUnit,
            typename TIdx,
            typename TWorkDiv>
        ALPAKA_FN_HOST_ACC auto getIdx(
            TIdx const & idx,
            TWorkDiv const & workDiv)
        -> vec::Vec<dim::Dim<TWorkDiv>, idx::Idx<TIdx>>
        {
            return
                traits::GetIdx<
                    TIdx,
                    TOrigin,
                    TUnit>
                ::getIdx(
                    idx,
                    workDiv);
        }
        //-----------------------------------------------------------------------------
        //! Get the indices requested.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOrigin,
            typename TUnit,
            typename TIdxWorkDiv>
        ALPAKA_FN_HOST_ACC auto getIdx(
            TIdxWorkDiv const & idxWorkDiv)
        -> vec::Vec<dim::Dim<TIdxWorkDiv>, idx::Idx<TIdxWorkDiv>>
        {
            return
                traits::GetIdx<
                    TIdxWorkDiv,
                    TOrigin,
                    TUnit>
                ::getIdx(
                    idxWorkDiv,
                    idxWorkDiv);
        }

        namespace traits
        {
            //#############################################################################
            //! The grid block index get trait specialization for classes with IdxGbBase member type.
            template<
                typename TIdxGb>
            struct GetIdx<
                TIdxGb,
                origin::Grid,
                unit::Blocks,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TIdxGb::IdxGbBase,
                        TIdxGb
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the grid.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST_ACC static auto getIdx(
                    TIdxGb const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<dim::Dim<typename TIdxGb::IdxGbBase>, idx::Idx<typename TIdxGb::IdxGbBase>>
                {
                    // Delegate the call to the base class.
                    return
                        idx::getIdx<
                            origin::Grid,
                            unit::Blocks>(
                                static_cast<typename TIdxGb::IdxGbBase const &>(idx),
                                workDiv);
                }
            };

            //#############################################################################
            //! The block thread index get trait specialization for classes with IdxBtBase member type.
            template<
                typename TIdxBt>
            struct GetIdx<
                TIdxBt,
                origin::Block,
                unit::Threads,
                typename std::enable_if<
                    meta::IsStrictBase<
                        typename TIdxBt::IdxBtBase,
                        TIdxBt
                    >::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the grid.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST_ACC static auto getIdx(
                    TIdxBt const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<dim::Dim<typename TIdxBt::IdxBtBase>, idx::Idx<typename TIdxBt::IdxBtBase>>
                {
                    // Delegate the call to the base class.
                    return
                        idx::getIdx<
                            origin::Block,
                            unit::Threads>(
                                static_cast<typename TIdxBt::IdxBtBase const &>(idx),
                                workDiv);
                }
            };

            //#############################################################################
            //! The grid thread index get trait specialization.
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Grid,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the grid.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST_ACC static auto getIdx(
                    TIdx const & idx,
                    TWorkDiv const & workDiv)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(
                    idx::getIdx<origin::Grid, unit::Blocks>(idx, workDiv)
                    * workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                    + idx::getIdx<origin::Block, unit::Threads>(idx, workDiv))
#endif
                {
                    return
                        idx::getIdx<origin::Grid, unit::Blocks>(idx, workDiv)
                        * workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                        + idx::getIdx<origin::Block, unit::Threads>(idx, workDiv);
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! Get the index of the first element this thread computes.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TIdxWorkDiv,
            typename TGridThreadIdx,
            typename TThreadElemExtent>
        ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(
            TIdxWorkDiv const & idxWorkDiv,
            TGridThreadIdx const & gridThreadIdx,
            TThreadElemExtent const & threadElemExtent)
        -> vec::Vec<dim::Dim<TIdxWorkDiv>, idx::Idx<TIdxWorkDiv>>
        {
            alpaka::ignore_unused(idxWorkDiv);

            return gridThreadIdx * threadElemExtent;
        }
        //-----------------------------------------------------------------------------
        //! Get the index of the first element this thread computes.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TIdxWorkDiv,
            typename TGridThreadIdx>
        ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(
            TIdxWorkDiv const & idxWorkDiv,
            TGridThreadIdx const & gridThreadIdx)
        -> vec::Vec<dim::Dim<TIdxWorkDiv>, idx::Idx<TIdxWorkDiv>>
        {
            auto const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(idxWorkDiv));
            return getIdxThreadFirstElem(idxWorkDiv, gridThreadIdx, threadElemExtent);
        }
        //-----------------------------------------------------------------------------
        //! Get the index of the first element this thread computes.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TIdxWorkDiv>
        ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(
            TIdxWorkDiv const & idxWorkDiv)
        -> vec::Vec<dim::Dim<TIdxWorkDiv>, idx::Idx<TIdxWorkDiv>>
        {
            auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(idxWorkDiv));
            return getIdxThreadFirstElem(idxWorkDiv, gridThreadIdx);
        }
    }
}
