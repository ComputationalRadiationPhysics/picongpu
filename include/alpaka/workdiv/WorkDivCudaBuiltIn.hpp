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

#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

#include <alpaka/core/Cuda.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The GPU CUDA accelerator work division.
        template<
            typename TDim,
            typename TIdx>
        class WorkDivCudaBuiltIn
        {
        public:
            using WorkDivBase = WorkDivCudaBuiltIn;

            //-----------------------------------------------------------------------------
            __device__ WorkDivCudaBuiltIn(
                vec::Vec<TDim, TIdx> const & threadElemExtent) :
                    m_threadElemExtent(threadElemExtent)
            {}
            //-----------------------------------------------------------------------------
            __device__ WorkDivCudaBuiltIn(WorkDivCudaBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ WorkDivCudaBuiltIn(WorkDivCudaBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(WorkDivCudaBuiltIn const &) -> WorkDivCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(WorkDivCudaBuiltIn &&) -> WorkDivCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~WorkDivCudaBuiltIn() = default;

        public:
            // \TODO: Optimize! Add WorkDivCudaBuiltInNoElems that has no member m_threadElemExtent as well as AccGpuCudaRtNoElems.
            // Use it instead of AccGpuCudaRt if the thread element extent is one to reduce the register usage.
            vec::Vec<TDim, TIdx> const & m_threadElemExtent;
        };
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                workdiv::WorkDivCudaBuiltIn<TDim, TIdx>>
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
            //! The GPU CUDA accelerator work division idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                workdiv::WorkDivCudaBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
    namespace workdiv
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division grid block extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivCudaBuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                __device__ static auto getWorkDiv(
                    WorkDivCudaBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(workDiv);
                    return vec::cast<TIdx>(extent::getExtentVecEnd<TDim>(gridDim));
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division block thread extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivCudaBuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                __device__ static auto getWorkDiv(
                    WorkDivCudaBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(workDiv);
                    return vec::cast<TIdx>(extent::getExtentVecEnd<TDim>(blockDim));
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division thread element extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivCudaBuiltIn<TDim, TIdx>,
                origin::Thread,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                __device__ static auto getWorkDiv(
                    WorkDivCudaBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_threadElemExtent;
                }
            };
        }
    }
}

#endif
