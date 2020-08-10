/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef _OPENMP

#include <alpaka/idx/Traits.hpp>
#include <alpaka/workdiv/Traits.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/idx/MapIdx.hpp>

#include <omp.h>


namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The OpenMP accelerator index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxBtOmp : public concepts::Implements<ConceptIdxBt, IdxBtOmp<TDim, TIdx>>
            {
            public:
                //-----------------------------------------------------------------------------
                IdxBtOmp() = default;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST IdxBtOmp(IdxBtOmp const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST IdxBtOmp(IdxBtOmp &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(IdxBtOmp const &) -> IdxBtOmp & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(IdxBtOmp &&) -> IdxBtOmp & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxBtOmp() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenMP accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::bt::IdxBtOmp<TDim, TIdx>>
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
            //! The OpenMP accelerator block thread index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::bt::IdxBtOmp<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST static auto getIdx(
                    idx::bt::IdxBtOmp<TDim, TIdx> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(idx);
                    // We assume that the thread id is positive.
                    ALPAKA_ASSERT(::omp_get_thread_num()>=0);
                    // \TODO: Would it be faster to precompute the index and cache it inside an array?
                    return idx::mapIdx<TDim::value>(
                        vec::Vec<dim::DimInt<1u>, TIdx>(static_cast<TIdx>(::omp_get_thread_num())),
                        workdiv::getWorkDiv<Block, Threads>(workDiv));
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenMP accelerator block thread index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtOmp<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
