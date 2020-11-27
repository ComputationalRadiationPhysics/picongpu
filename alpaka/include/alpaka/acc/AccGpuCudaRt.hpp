/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/BoostPredef.hpp>

#    if !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

// Base classes.
#    include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>

// Specialized traits.
#    include <alpaka/acc/Traits.hpp>

// Implementation details.
#    include <alpaka/core/ClipCast.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/Cuda.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>

#    include <typeinfo>

namespace alpaka
{
    template<typename TAcc, typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelGpuUniformCudaHipRt;

    //#############################################################################
    //! The GPU CUDA accelerator.
    //!
    //! This accelerator allows parallel kernel execution on devices supporting CUDA.
    template<typename TDim, typename TIdx>
    class AccGpuCudaRt final
        : public AccGpuUniformCudaHipRt<TDim, TIdx>
        , public concepts::Implements<ConceptUniformCudaHip, AccGpuUniformCudaHipRt<TDim, TIdx>>
    {
        static_assert(
            sizeof(TIdx) >= sizeof(int),
            "Index type is not supported, consider using int or a larger type.");

    public:
        //-----------------------------------------------------------------------------
        __device__ AccGpuCudaRt(Vec<TDim, TIdx> const& threadElemExtent)
            : AccGpuUniformCudaHipRt<TDim, TIdx>(threadElemExtent)
        {
        }

    public:
        //-----------------------------------------------------------------------------
        __device__ AccGpuCudaRt(AccGpuCudaRt const&) = delete;
        //-----------------------------------------------------------------------------
        __device__ AccGpuCudaRt(AccGpuCudaRt&&) = delete;
        //-----------------------------------------------------------------------------
        __device__ auto operator=(AccGpuCudaRt const&) -> AccGpuCudaRt& = delete;
        //-----------------------------------------------------------------------------
        __device__ auto operator=(AccGpuCudaRt&&) -> AccGpuCudaRt& = delete;
        //-----------------------------------------------------------------------------
        ~AccGpuCudaRt() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The GPU CUDA accelerator accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccGpuCudaRt<TDim, TIdx>>
        {
            using type = AccGpuCudaRt<TDim, TIdx>;
        };

        //#############################################################################
        //! The GPU CUDA accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccGpuCudaRt<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccGpuCudaRt<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The GPU CUDA accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccGpuCudaRt<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto createTaskKernel(
                TWorkDiv const& workDiv,
                TKernelFnObj const& kernelFnObj,
                TArgs&&... args)
            {
                return TaskKernelGpuUniformCudaHipRt<AccGpuCudaRt<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>(
                    workDiv,
                    kernelFnObj,
                    std::forward<TArgs>(args)...);
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
