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

#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

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
                typename TIdx>
            class IdxBtZero
            {
            public:
                using IdxBtBase = IdxBtZero;

                //-----------------------------------------------------------------------------
                IdxBtZero() = default;
                //-----------------------------------------------------------------------------
                IdxBtZero(IdxBtZero const &) = delete;
                //-----------------------------------------------------------------------------
                IdxBtZero(IdxBtZero &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxBtZero const &) -> IdxBtZero & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxBtZero &&) -> IdxBtZero & = delete;
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
                typename TIdx>
            struct DimType<
                idx::bt::IdxBtZero<TDim, TIdx>>
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
                typename TIdx>
            struct GetIdx<
                idx::bt::IdxBtZero<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST static auto getIdx(
                    idx::bt::IdxBtZero<TDim, TIdx> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(idx);
                    alpaka::ignore_unused(workDiv);
                    return vec::Vec<TDim, TIdx>::zeros();
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The zero block thread index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtZero<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}
