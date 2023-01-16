/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include <alpaka/acc/AccGenericSycl.hpp>
#    include <alpaka/acc/Tag.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/DevGpuSyclIntel.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/kernel/TaskKernelGpuSyclIntel.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/PltfGpuSyclIntel.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <CL/sycl.hpp>

#    include <string>
#    include <utility>

namespace alpaka::experimental
{
    //! The Intel GPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Intel GPU target device.
    template<typename TDim, typename TIdx>
    class AccGpuSyclIntel final
        : public AccGenericSycl<TDim, TIdx>
        , public concepts::Implements<ConceptAcc, AccGpuSyclIntel<TDim, TIdx>>
    {
    public:
        using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    //! The Intel GPU SYCL accelerator name trait specialization.
    template<typename TDim, typename TIdx>
    struct GetAccName<experimental::AccGpuSyclIntel<TDim, TIdx>>
    {
        static auto getAccName() -> std::string
        {
            return "experimental::AccGpuSyclIntel<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
        }
    };

    //! The Intel GPU SYCL accelerator device type trait specialization.
    template<typename TDim, typename TIdx>
    struct DevType<experimental::AccGpuSyclIntel<TDim, TIdx>>
    {
        using type = experimental::DevGpuSyclIntel;
    };

    //! The Intel GPU SYCL accelerator execution task type trait specialization.
    template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    struct CreateTaskKernel<experimental::AccGpuSyclIntel<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
    {
        static auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        {
            return experimental::TaskKernelGpuSyclIntel<TDim, TIdx, TKernelFnObj, TArgs...>{
                workDiv,
                kernelFnObj,
                std::forward<TArgs>(args)...};
        }
    };

    //! The Intel GPU SYCL execution task platform type trait specialization.
    template<typename TDim, typename TIdx>
    struct PltfType<experimental::AccGpuSyclIntel<TDim, TIdx>>
    {
        using type = experimental::PltfGpuSyclIntel;
    };

    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::experimental::AccGpuSyclIntel<TDim, TIdx>>
    {
        using type = alpaka::TagGpuSyclIntel;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<alpaka::TagGpuSyclIntel, TDim, TIdx>
    {
        using type = alpaka::experimental::AccGpuSyclIntel<TDim, TIdx>;
    };
} // namespace alpaka::trait

#endif
