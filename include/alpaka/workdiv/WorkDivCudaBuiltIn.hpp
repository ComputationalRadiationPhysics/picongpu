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
#include <alpaka/size/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Cuda.hpp>

//#include <boost/core/ignore_unused.hpp>

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The GPU CUDA accelerator work division.
        template<
            typename TDim,
            typename TSize>
        class WorkDivCudaBuiltIn
        {
        public:
            using WorkDivBase = WorkDivCudaBuiltIn;

            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY WorkDivCudaBuiltIn(
                vec::Vec<TDim, TSize> const & threadElemExtent) :
                    m_threadElemExtent(threadElemExtent)
            {}
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY WorkDivCudaBuiltIn(WorkDivCudaBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY WorkDivCudaBuiltIn(WorkDivCudaBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(WorkDivCudaBuiltIn const &) -> WorkDivCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(WorkDivCudaBuiltIn &&) -> WorkDivCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~WorkDivCudaBuiltIn() = default;

        public:
            // \TODO: Optimize! Add WorkDivCudaBuiltInNoElems that has no member m_threadElemExtent as well as AccGpuCudaRtNoElems.
            // Use it instead of AccGpuCudaRt if the thread element extent is one to reduce the register usage.
            vec::Vec<TDim, TSize> const & m_threadElemExtent;
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
                typename TSize>
            struct DimType<
                workdiv::WorkDivCudaBuiltIn<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division size type trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                workdiv::WorkDivCudaBuiltIn<TDim, TSize>>
            {
                using type = TSize;
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
                typename TSize>
            struct GetWorkDiv<
                WorkDivCudaBuiltIn<TDim, TSize>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                ALPAKA_FN_ACC_CUDA_ONLY static auto getWorkDiv(
                    WorkDivCudaBuiltIn<TDim, TSize> const & /*workDiv*/)
                -> vec::Vec<TDim, TSize>
                {
                    //boost::ignore_unused(workDiv);
                    return vec::cast<TSize>(extent::getExtentVecEnd<TDim>(gridDim));
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division block thread extent trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivCudaBuiltIn<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                ALPAKA_FN_ACC_CUDA_ONLY static auto getWorkDiv(
                    WorkDivCudaBuiltIn<TDim, TSize> const & /*workDiv*/)
                -> vec::Vec<TDim, TSize>
                {
                    //boost::ignore_unused(workDiv);
                    return vec::cast<TSize>(extent::getExtentVecEnd<TDim>(blockDim));
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division thread element extent trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivCudaBuiltIn<TDim, TSize>,
                origin::Thread,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                ALPAKA_FN_ACC_CUDA_ONLY static auto getWorkDiv(
                    WorkDivCudaBuiltIn<TDim, TSize> const & workDiv)
                -> vec::Vec<TDim, TSize>
                {
                    return workDiv.m_threadElemExtent;
                }
            };
        }
    }
}

#endif
