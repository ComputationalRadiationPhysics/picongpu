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

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

#include <alpaka/core/Hip.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

#include <hip/hip_runtime.h>


namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The GPU HIP accelerator work division.
        template<
            typename TDim,
            typename TIdx>
        class WorkDivHipBuiltIn
        {
        public:
            using WorkDivBase = WorkDivHipBuiltIn;

            //-----------------------------------------------------------------------------
            //! Default constructor.
            __device__ WorkDivHipBuiltIn(
                vec::Vec<TDim, TIdx> const & threadElemExtent) :
                    m_threadElemExtent(threadElemExtent)
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            __device__ WorkDivHipBuiltIn(WorkDivHipBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            __device__ WorkDivHipBuiltIn(WorkDivHipBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            __device__ auto operator=(WorkDivHipBuiltIn const &) -> WorkDivHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            __device__ auto operator=(WorkDivHipBuiltIn &&) -> WorkDivHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            /*virtual*/ ALPAKA_FN_HOST_ACC ~WorkDivHipBuiltIn() = default;

        public:
            // \TODO: Optimize! Add WorkDivHipBuiltInNoElems that has no member m_threadElemExtent as well as AccGpuHipRtNoElems.
            // Use it instead of AccGpuHipRt if the thread element extent is one to reduce the register usage.
            vec::Vec<TDim, TIdx> const & m_threadElemExtent;
        };
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator work division dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                workdiv::WorkDivHipBuiltIn<TDim, TIdx>>
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
            //! The GPU HIP accelerator work division idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                workdiv::WorkDivHipBuiltIn<TDim, TIdx>>
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
            //! The GPU HIP accelerator work division grid block extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivHipBuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                ALPAKA_NO_HOST_ACC_WARNING
#if defined(BOOST_COMP_HCC) && BOOST_COMP_HCC /* hcc requires matching host-device signature */
                ALPAKA_FN_HOST_ACC
#else /* nvcc does not know about blockDim.x etc. on host */
                __device__
#endif
                static auto getWorkDiv(
                    WorkDivHipBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(workDiv);

                    return extent::getExtentVecEnd<TDim>(
                        vec::Vec<
                          std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                            static_cast<TIdx>(hipGridDim_z),
                            static_cast<TIdx>(hipGridDim_y),
                            static_cast<TIdx>(hipGridDim_x)));
                }
            };

            //#############################################################################
            //! The GPU HIP accelerator work division block thread extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivHipBuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                ALPAKA_NO_HOST_ACC_WARNING
#if defined(BOOST_COMP_HCC) && BOOST_COMP_HCC /* hcc requires matching host-device signature */
                ALPAKA_FN_HOST_ACC
#else /* nvcc does not know about blockDim.x etc. on host */
                __device__
#endif
                static auto getWorkDiv(
                    WorkDivHipBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(workDiv);

                    return extent::getExtentVecEnd<TDim>(
                        vec::Vec<
                          std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                            static_cast<TIdx>(hipBlockDim_z),
                            static_cast<TIdx>(hipBlockDim_y),
                            static_cast<TIdx>(hipBlockDim_x)));
                }
            };

            //#############################################################################
            //! The GPU HIP accelerator work division thread element extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivHipBuiltIn<TDim, TIdx>,
                origin::Thread,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    WorkDivHipBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_threadElemExtent;
                }
            };
        }
    }
}

#endif
