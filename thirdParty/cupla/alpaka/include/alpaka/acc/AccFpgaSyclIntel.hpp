/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

#    include <alpaka/acc/AccGenericSycl.hpp>
#    include <alpaka/acc/Tag.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/DevFpgaSyclIntel.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/kernel/TaskKernelFpgaSyclIntel.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/PltfFpgaSyclIntel.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <CL/sycl.hpp>

#    include <string>
#    include <utility>

namespace alpaka::experimental
{
    //! The Intel FPGA SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Intel FPGA target device.
    template<typename TDim, typename TIdx>
    class AccFpgaSyclIntel final
        : public AccGenericSycl<TDim, TIdx>
        , public concepts::Implements<ConceptAcc, AccFpgaSyclIntel<TDim, TIdx>>
    {
    public:
        using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    //! The Intel FPGA SYCL accelerator name trait specialization.
    template<typename TDim, typename TIdx>
    struct GetAccName<experimental::AccFpgaSyclIntel<TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getAccName() -> std::string
        {
            return "experimental::AccFpgaSyclIntel<" + std::to_string(TDim::value) + "," + core::demangled<TIdx> + ">";
        }
    };

    //! The Intel FPGA SYCL accelerator device type trait specialization.
    template<typename TDim, typename TIdx>
    struct DevType<experimental::AccFpgaSyclIntel<TDim, TIdx>>
    {
        using type = experimental::DevFpgaSyclIntel;
    };

    //! The Intel FPGA SYCL accelerator execution task type trait specialization.
    template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    struct CreateTaskKernel<experimental::AccFpgaSyclIntel<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
    {
        static auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        {
            return experimental::TaskKernelFpgaSyclIntel<TDim, TIdx, TKernelFnObj, TArgs...>{
                workDiv,
                kernelFnObj,
                std::forward<TArgs>(args)...};
        }
    };

    //! The Intel FPGA SYCL execution task platform type trait specialization.
    template<typename TDim, typename TIdx>
    struct PltfType<experimental::AccFpgaSyclIntel<TDim, TIdx>>
    {
        using type = experimental::PltfFpgaSyclIntel;
    };

    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::experimental::AccFpgaSyclIntel<TDim, TIdx>>
    {
        using type = alpaka::TagFpgaSyclIntel;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<alpaka::TagFpgaSyclIntel, TDim, TIdx>
    {
        using type = alpaka::experimental::AccFpgaSyclIntel<TDim, TIdx>;
    };
} // namespace alpaka::trait

#endif
