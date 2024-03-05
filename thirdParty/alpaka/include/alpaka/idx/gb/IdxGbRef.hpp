/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

namespace alpaka
{
    namespace gb
    {
        //! A IdxGbRef grid block index.
        template<typename TDim, typename TIdx>
        class IdxGbRef : public concepts::Implements<ConceptIdxGb, IdxGbRef<TDim, TIdx>>
        {
        public:
            IdxGbRef(Vec<TDim, TIdx> const& gridBlockIdx) : m_gridBlockIdx(gridBlockIdx)
            {
            }

            Vec<TDim, TIdx> const& m_gridBlockIdx;
        };
    } // namespace gb

    namespace trait
    {
        //! The IdxGbRef grid block index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<gb::IdxGbRef<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The IdxGbRef grid block index grid block index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<gb::IdxGbRef<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            ALPAKA_FN_HOST static auto getIdx(gb::IdxGbRef<TDim, TIdx> const& idx, TWorkDiv const& /* workDiv */)
                -> Vec<TDim, TIdx>
            {
                return idx.m_gridBlockIdx;
            }
        };

        //! The IdxGbRef grid block index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<gb::IdxGbRef<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka
