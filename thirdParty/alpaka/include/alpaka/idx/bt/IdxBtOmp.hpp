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

#ifdef _OPENMP

#include <alpaka/idx/Traits.hpp>
#include <alpaka/workdiv/Traits.hpp>

#include <alpaka/core/Assert.hpp>
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
            class IdxBtOmp
            {
            public:
                using IdxBtBase = IdxBtOmp;

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
