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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Cuda.hpp>

//#include <boost/core/ignore_unused.hpp>

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The CUDA accelerator ND index provider.
            template<
                typename TDim,
                typename TSize>
            class IdxBtCudaBuiltIn
            {
            public:
                using IdxBtBase = IdxBtCudaBuiltIn;

                //-----------------------------------------------------------------------------
                IdxBtCudaBuiltIn() = default;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY IdxBtCudaBuiltIn(IdxBtCudaBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY IdxBtCudaBuiltIn(IdxBtCudaBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY auto operator=(IdxBtCudaBuiltIn const & ) -> IdxBtCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY auto operator=(IdxBtCudaBuiltIn &&) -> IdxBtCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxBtCudaBuiltIn() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                idx::bt::IdxBtCudaBuiltIn<TDim, TSize>>
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
            //! The GPU CUDA accelerator block thread index get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetIdx<
                idx::bt::IdxBtCudaBuiltIn<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_CUDA_ONLY static auto getIdx(
                    idx::bt::IdxBtCudaBuiltIn<TDim, TSize> const & /*idx*/,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TSize>
                {
                    //boost::ignore_unused(idx);
                    return vec::cast<TSize>(offset::getOffsetVecEnd<TDim>(threadIdx));
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator block thread index size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                idx::bt::IdxBtCudaBuiltIn<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
