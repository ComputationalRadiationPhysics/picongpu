/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"
#include "alpaka/workdiv/Traits.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    //! The GPU CUDA/HIP accelerator work division.
    template<typename TDim, typename TIdx>
    class WorkDivUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptWorkDiv, WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
    {
    public:
        ALPAKA_FN_HOST_ACC WorkDivUniformCudaHipBuiltIn(Vec<TDim, TIdx> const& threadElemExtent)
            : m_threadElemExtent(threadElemExtent)
        {
        }

        // \TODO: Optimize! Add WorkDivUniformCudaHipBuiltInNoElems that has no member m_threadElemExtent as well as
        // AccGpuUniformCudaHipRtNoElems. Use it instead of AccGpuUniformCudaHipRt if the thread element extent is one
        // to reduce the register usage.
        Vec<TDim, TIdx> const& m_threadElemExtent;
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        //! The GPU CUDA/HIP accelerator work division dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The GPU CUDA/HIP accelerator work division idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The GPU CUDA/HIP accelerator work division grid block extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The number of blocks in each dimension of the grid.
            __device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& /* workDiv */)
                -> Vec<TDim, TIdx>
            {
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                return castVec<TIdx>(getExtentVecEnd<TDim>(gridDim));
#        else
                return getExtentVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipGridDim_z),
                    static_cast<TIdx>(hipGridDim_y),
                    static_cast<TIdx>(hipGridDim_x)));
#        endif
            }
        };

        //! The GPU CUDA/HIP accelerator work division block thread extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The number of threads in each dimension of a block.
            __device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& /* workDiv */)
                -> Vec<TDim, TIdx>
            {
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                return castVec<TIdx>(getExtentVecEnd<TDim>(blockDim));
#        else
                return getExtentVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipBlockDim_z),
                    static_cast<TIdx>(hipBlockDim_y),
                    static_cast<TIdx>(hipBlockDim_x)));
#        endif
            }
        };

        //! The GPU CUDA/HIP accelerator work division thread element extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Thread, unit::Elems>
        {
            //! \return The number of blocks in each dimension of the grid.
            __device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& workDiv)
                -> Vec<TDim, TIdx>
            {
                return workDiv.m_threadElemExtent;
            }
        };
    } // namespace trait

#    endif

} // namespace alpaka

#endif
