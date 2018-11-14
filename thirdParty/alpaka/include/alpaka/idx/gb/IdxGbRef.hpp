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

#include <alpaka/idx/Traits.hpp>
#include <alpaka/dim/Traits.hpp>

#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

namespace alpaka
{
    namespace idx
    {
        namespace gb
        {
            //#############################################################################
            //! A IdxGbRef grid block index.
            template<
                typename TDim,
                typename TIdx>
            class IdxGbRef
            {
            public:
                using IdxGbBase = IdxGbRef;

                //-----------------------------------------------------------------------------
                IdxGbRef(
                    vec::Vec<TDim, TIdx> const & gridBlockIdx) :
                        m_gridBlockIdx(gridBlockIdx)
                {}
                //-----------------------------------------------------------------------------
                IdxGbRef(IdxGbRef const &) = delete;
                //-----------------------------------------------------------------------------
                IdxGbRef(IdxGbRef &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbRef const &) -> IdxGbRef & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbRef &&) -> IdxGbRef & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxGbRef() = default;

            public:
                vec::Vec<TDim, TIdx> const & m_gridBlockIdx;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The IdxGbRef grid block index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::gb::IdxGbRef<TDim, TIdx>>
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
            //! The IdxGbRef grid block index grid block index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::gb::IdxGbRef<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST static auto getIdx(
                    idx::gb::IdxGbRef<TDim, TIdx> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(workDiv);
                    return idx.m_gridBlockIdx;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The IdxGbRef grid block index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::gb::IdxGbRef<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}
