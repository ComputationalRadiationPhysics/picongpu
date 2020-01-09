/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Cuda.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>

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
                typename TIdx>
            class IdxBtCudaBuiltIn
            {
            public:
                using IdxBtBase = IdxBtCudaBuiltIn;

                //-----------------------------------------------------------------------------
                IdxBtCudaBuiltIn() = default;
                //-----------------------------------------------------------------------------
                __device__ IdxBtCudaBuiltIn(IdxBtCudaBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                __device__ IdxBtCudaBuiltIn(IdxBtCudaBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(IdxBtCudaBuiltIn const & ) -> IdxBtCudaBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(IdxBtCudaBuiltIn &&) -> IdxBtCudaBuiltIn & = delete;
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
                typename TIdx>
            struct DimType<
                idx::bt::IdxBtCudaBuiltIn<TDim, TIdx>>
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
                typename TIdx>
            struct GetIdx<
                idx::bt::IdxBtCudaBuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                __device__ static auto getIdx(
                    idx::bt::IdxBtCudaBuiltIn<TDim, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(idx);
                    return vec::cast<TIdx>(offset::getOffsetVecEnd<TDim>(threadIdx));
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator block thread index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtCudaBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
