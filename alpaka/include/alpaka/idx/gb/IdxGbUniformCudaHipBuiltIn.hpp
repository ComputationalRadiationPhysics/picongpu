/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <alpaka/core/Cuda.hpp>
#else
#include <alpaka/core/Hip.hpp>
#endif

namespace alpaka
{
    namespace idx
    {
        namespace gb
        {
            //#############################################################################
            //! The CUDA/HIP accelerator ND index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxGbUniformCudaHipBuiltIn : public concepts::Implements<ConceptIdxGb, IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
            {
            public:
                //-----------------------------------------------------------------------------
                IdxGbUniformCudaHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                __device__ IdxGbUniformCudaHipBuiltIn(IdxGbUniformCudaHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                __device__ IdxGbUniformCudaHipBuiltIn(IdxGbUniformCudaHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(IdxGbUniformCudaHipBuiltIn const & ) -> IdxGbUniformCudaHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                __device__ auto operator=(IdxGbUniformCudaHipBuiltIn &&) -> IdxGbUniformCudaHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxGbUniformCudaHipBuiltIn() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA/HIP accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
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
            //! The GPU CUDA/HIP accelerator grid block index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                __device__ static auto getIdx(
                    idx::gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(idx);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    return vec::cast<TIdx>(offset::getOffsetVecEnd<TDim>(blockIdx));
#else
                    return offset::getOffsetVecEnd<TDim>(
                        vec::Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                            static_cast<TIdx>(hipBlockIdx_z),
                            static_cast<TIdx>(hipBlockIdx_y),
                            static_cast<TIdx>(hipBlockIdx_x)));
#endif
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA/HIP accelerator grid block index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
