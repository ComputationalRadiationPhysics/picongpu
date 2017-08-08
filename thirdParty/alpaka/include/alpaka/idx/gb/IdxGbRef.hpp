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

#include <alpaka/idx/Traits.hpp>            // idx::getIdx

#include <alpaka/dim/Traits.hpp>            // dim::Dim

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

namespace alpaka
{
    namespace idx
    {
        namespace gb
        {
            //#############################################################################
            //! A IdxGbRef grid block index.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            class IdxGbRef
            {
            public:
                using IdxGbBase = IdxGbRef;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxGbRef(
                    vec::Vec<TDim, TSize> const & gridBlockIdx) :
                        m_gridBlockIdx(gridBlockIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxGbRef(IdxGbRef const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxGbRef(IdxGbRef &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxGbRef const &) -> IdxGbRef & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxGbRef &&) -> IdxGbRef & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~IdxGbRef() = default;

            public:
                vec::Vec<TDim, TSize> const & m_gridBlockIdx;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The IdxGbRef grid block index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                idx::gb::IdxGbRef<TDim, TSize>>
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
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetIdx<
                idx::gb::IdxGbRef<TDim, TSize>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_NO_CUDA static auto getIdx(
                    idx::gb::IdxGbRef<TDim, TSize> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TSize>
                {
                    boost::ignore_unused(workDiv);
                    return idx.m_gridBlockIdx;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The IdxGbRef grid block index size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                idx::gb::IdxGbRef<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
