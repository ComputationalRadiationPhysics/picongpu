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

#include <alpaka/idx/MapIdx.hpp>

#include <alpaka/core/OpenMp.hpp>

#include <boost/core/ignore_unused.hpp>

#include <cassert>

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
                typename TSize>
            class IdxBtOmp
            {
            public:
                using IdxBtBase = IdxBtOmp;

                //-----------------------------------------------------------------------------
                IdxBtOmp() = default;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtOmp(IdxBtOmp const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtOmp(IdxBtOmp &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtOmp const &) -> IdxBtOmp & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtOmp &&) -> IdxBtOmp & = delete;
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
                typename TSize>
            struct DimType<
                idx::bt::IdxBtOmp<TDim, TSize>>
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
                typename TSize>
            struct GetIdx<
                idx::bt::IdxBtOmp<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_NO_CUDA static auto getIdx(
                    idx::bt::IdxBtOmp<TDim, TSize> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TSize>
                {
                    boost::ignore_unused(idx);
                    // We assume that the thread id is positive.
                    assert(::omp_get_thread_num()>=0);
                    // \TODO: Would it be faster to precompute the index and cache it inside an array?
                    return idx::mapIdx<TDim::value>(
                        vec::Vec<dim::DimInt<1u>, TSize>(static_cast<TSize>(::omp_get_thread_num())),
                        workdiv::getWorkDiv<Block, Threads>(workDiv));
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenMP accelerator block thread index size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                idx::bt::IdxBtOmp<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
