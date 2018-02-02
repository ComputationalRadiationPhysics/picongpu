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

#include <boost/core/ignore_unused.hpp>

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! A zero block thread index provider.
            template<
                typename TDim,
                typename TSize>
            class IdxBtZero
            {
            public:
                using IdxBtBase = IdxBtZero;

                //-----------------------------------------------------------------------------
                IdxBtZero() = default;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtZero(IdxBtZero const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtZero(IdxBtZero &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtZero const &) -> IdxBtZero & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtZero &&) -> IdxBtZero & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxBtZero() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The zero block thread index provider dimension get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                idx::bt::IdxBtZero<TDim, TSize>>
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
            //! The zero block thread index provider block thread index get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetIdx<
                idx::bt::IdxBtZero<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_NO_CUDA static auto getIdx(
                    idx::bt::IdxBtZero<TDim, TSize> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TSize>
                {
                    boost::ignore_unused(idx);
                    boost::ignore_unused(workDiv);
                    return vec::Vec<TDim, TSize>::zeros();
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The zero block thread index size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                idx::bt::IdxBtZero<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
