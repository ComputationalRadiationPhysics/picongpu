/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevCpuSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/kernel/TaskKernelCpuSycl.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/PlatformCpuSycl.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <string>
#include <utility>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)

namespace alpaka
{
    //! The CPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable CPU target device.
    template<typename TDim, typename TIdx>
    class AccCpuSycl final
        : public AccGenericSycl<TDim, TIdx>
        , public concepts::Implements<ConceptAcc, AccCpuSycl<TDim, TIdx>>
    {
    public:
        using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The CPU SYCL accelerator name trait specialization.
    template<typename TDim, typename TIdx>
    struct GetAccName<AccCpuSycl<TDim, TIdx>>
    {
        static auto getAccName() -> std::string
        {
            return "AccCpuSycl<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
        }
    };

    //! The CPU SYCL accelerator device type trait specialization.
    template<typename TDim, typename TIdx>
    struct DevType<AccCpuSycl<TDim, TIdx>>
    {
        using type = DevCpuSycl;
    };

    //! The CPU SYCL accelerator execution task type trait specialization.
    template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    struct CreateTaskKernel<AccCpuSycl<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
    {
        static auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        {
            return TaskKernelCpuSycl<TDim, TIdx, TKernelFnObj, TArgs...>{
                workDiv,
                kernelFnObj,
                std::forward<TArgs>(args)...};
        }
    };

    //! The CPU SYCL execution task platform type trait specialization.
    template<typename TDim, typename TIdx>
    struct PlatformType<AccCpuSycl<TDim, TIdx>>
    {
        using type = PlatformCpuSycl;
    };

    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::AccCpuSycl<TDim, TIdx>>
    {
        using type = alpaka::TagCpuSycl;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<alpaka::TagCpuSycl, TDim, TIdx>
    {
        using type = alpaka::AccCpuSycl<TDim, TIdx>;
    };
} // namespace alpaka::trait

#endif
