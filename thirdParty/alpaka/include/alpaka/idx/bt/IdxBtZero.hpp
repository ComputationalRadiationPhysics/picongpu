/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

namespace alpaka
{
    namespace bt
    {
        //! A zero block thread index provider.
        template<typename TDim, typename TIdx>
        class IdxBtZero : public concepts::Implements<ConceptIdxBt, IdxBtZero<TDim, TIdx>>
        {
        };
    } // namespace bt

    namespace trait
    {
        //! The zero block thread index provider dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<bt::IdxBtZero<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The zero block thread index provider block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtZero<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            ALPAKA_FN_HOST static auto getIdx(
                bt::IdxBtZero<TDim, TIdx> const& /* idx */,
                TWorkDiv const& /* workDiv */) -> Vec<TDim, TIdx>
            {
                return Vec<TDim, TIdx>::zeros();
            }
        };

        //! The zero block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtZero<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka
