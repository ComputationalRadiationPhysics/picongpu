/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/idx/Traits.hpp>

#include <alpaka/core/Concepts.hpp>
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
            class IdxBtZero : public concepts::Implements<ConceptIdxBt, IdxBtZero<TDim, TIdx>>
            {
            public:
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
