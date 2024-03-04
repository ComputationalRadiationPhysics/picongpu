/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Matthias Werner, Jan Stephan, Andrea Bocci, Bernhard
 * Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    namespace gb
    {
        //! The CUDA/HIP accelerator ND index provider.
        template<typename TDim, typename TIdx>
        class IdxGbUniformCudaHipBuiltIn
            : public concepts::Implements<ConceptIdxGb, IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
        {
        };
    } // namespace gb

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        //! The GPU CUDA/HIP accelerator index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The GPU CUDA/HIP accelerator grid block index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            __device__ static auto getIdx(gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx> const& /* idx */, TWorkDiv const&)
                -> Vec<TDim, TIdx>
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return castVec<TIdx>(getOffsetVecEnd<TDim>(blockIdx));
#        else
                return getOffsetVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipBlockIdx_z),
                    static_cast<TIdx>(hipBlockIdx_y),
                    static_cast<TIdx>(hipBlockIdx_x)));
#        endif
            }
        };

        //! The GPU CUDA/HIP accelerator grid block index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait

#    endif

} // namespace alpaka

#endif
