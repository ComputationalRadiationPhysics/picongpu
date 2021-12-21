/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

namespace alpaka
{
    namespace gb
    {
        //#############################################################################
        //! A IdxGbRef grid block index.
        template<typename TDim, typename TIdx>
        class IdxGbRef : public concepts::Implements<ConceptIdxGb, IdxGbRef<TDim, TIdx>>
        {
        public:
            //-----------------------------------------------------------------------------
            IdxGbRef(Vec<TDim, TIdx> const& gridBlockIdx) : m_gridBlockIdx(gridBlockIdx)
            {
            }
            //-----------------------------------------------------------------------------
            IdxGbRef(IdxGbRef const&) = delete;
            //-----------------------------------------------------------------------------
            IdxGbRef(IdxGbRef&&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxGbRef const&) -> IdxGbRef& = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxGbRef&&) -> IdxGbRef& = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~IdxGbRef() = default;

        public:
            Vec<TDim, TIdx> const& m_gridBlockIdx;
        };
    } // namespace gb

    namespace traits
    {
        //#############################################################################
        //! The IdxGbRef grid block index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<gb::IdxGbRef<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The IdxGbRef grid block index grid block index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<gb::IdxGbRef<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            ALPAKA_FN_HOST static auto getIdx(gb::IdxGbRef<TDim, TIdx> const& idx, TWorkDiv const& workDiv)
                -> Vec<TDim, TIdx>
            {
                alpaka::ignore_unused(workDiv);
                return idx.m_gridBlockIdx;
            }
        };

        //#############################################################################
        //! The IdxGbRef grid block index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<gb::IdxGbRef<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka
