/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Andrea Bocci, Bernhard
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
    namespace bt
    {
        //! The CUDA/HIP accelerator ND index provider.
        template<typename TDim, typename TIdx>
        class IdxBtUniformCudaHipBuiltIn
            : public concepts::Implements<ConceptIdxBt, IdxBtUniformCudaHipBuiltIn<TDim, TIdx>>
        {
        };
    } // namespace bt

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
        struct DimType<bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The GPU CUDA/HIP accelerator block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            __device__ static auto getIdx(bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx> const& /* idx */, TWorkDiv const&)
                -> Vec<TDim, TIdx>
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return castVec<TIdx>(getOffsetVecEnd<TDim>(threadIdx));
#        else
                return getOffsetVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipThreadIdx_z),
                    static_cast<TIdx>(hipThreadIdx_y),
                    static_cast<TIdx>(hipThreadIdx_x)));
#        endif
            }
        };

        //! The GPU CUDA/HIP accelerator block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait

#    endif

} // namespace alpaka

#endif
