/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/BoostPredef.hpp>

#    if !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

// Base classes.
#    include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>

// Implementation details.
#    include <alpaka/core/ClipCast.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Hip.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>

#    include <typeinfo>

namespace alpaka
{
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelGpuUniformCudaHipRt;

    //#############################################################################
    //! The GPU HIP accelerator.
    //!
    //! This accelerator allows parallel kernel execution on devices supporting HIP
    template<typename TDim, typename TIdx>
    class AccGpuHipRt final
        : public AccGpuUniformCudaHipRt<TDim, TIdx>
        , public concepts::Implements<ConceptUniformCudaHip, AccGpuUniformCudaHipRt<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        //-----------------------------------------------------------------------------
        __device__ AccGpuHipRt(Vec<TDim, TIdx> const& threadElemExtent)
            : AccGpuUniformCudaHipRt<TDim, TIdx>(threadElemExtent)
        {
        }

    public:
        //-----------------------------------------------------------------------------
        __device__ AccGpuHipRt(AccGpuHipRt const&) = delete;
        //-----------------------------------------------------------------------------
        __device__ AccGpuHipRt(AccGpuHipRt&&) = delete;
        //-----------------------------------------------------------------------------
        __device__ auto operator=(AccGpuHipRt const&) -> AccGpuHipRt& = delete;
        //-----------------------------------------------------------------------------
        __device__ auto operator=(AccGpuHipRt&&) -> AccGpuHipRt& = delete;
        //-----------------------------------------------------------------------------
        ~AccGpuHipRt() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The GPU HIP accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccGpuHipRt<TDim, TIdx>>
        {
            using type = AccGpuHipRt<TDim, TIdx>;
        };

        //#############################################################################
        //! The GPU Hip accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccGpuHipRt<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccGpuHipRt<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The GPU HIP accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccGpuHipRt<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelGpuUniformCudaHipRt<AccGpuHipRt<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
