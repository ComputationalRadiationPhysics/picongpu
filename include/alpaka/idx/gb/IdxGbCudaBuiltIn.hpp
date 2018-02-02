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
        namespace gb
        {
            //#############################################################################
            //! The CUDA accelerator ND index provider.
            template<
                typename TDim,
                typename TSize>
            class IdxGbCudaBuiltIn
            {
            public:
                using IdxGbBase = IdxGbCudaBuiltIn;

                //-----------------------------------------------------------------------------
                IdxGbCudaBuiltIn() = default;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY IdxGbCudaBuiltIn(IdxGbCudaBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY IdxGbCudaBuiltIn(IdxGbCudaBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY auto operator=(IdxGbCudaBuiltIn const & ) -> IdxGbCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY auto operator=(IdxGbCudaBuiltIn &&) -> IdxGbCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxGbCudaBuiltIn() = default;
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
                idx::gb::IdxGbCudaBuiltIn<TDim, TSize>>
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
            //! The GPU CUDA accelerator grid block index get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetIdx<
                idx::gb::IdxGbCudaBuiltIn<TDim, TSize>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_CUDA_ONLY static auto getIdx(
                    idx::gb::IdxGbCudaBuiltIn<TDim, TSize> const & /*idx*/,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TSize>
                {
                    //boost::ignore_unused(idx);
                    return vec::cast<TSize>(offset::getOffsetVecEnd<TDim>(blockIdx));
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator grid block index size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                idx::gb::IdxGbCudaBuiltIn<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#endif
