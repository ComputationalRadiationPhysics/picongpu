/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <alpaka/acc/AccGenericSycl.hpp>
#    include <alpaka/acc/Tag.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/DemangleTypeNames.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/kernel/TaskKernelFpgaSyclXilinx.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/pltf/PltfFpgaSyclXilinx.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <CL/sycl.hpp>

#    include <string>
#    include <utility>

namespace alpaka::experimental
{
    //! The Xilinx FPGA SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a SYCL-capable Xilinx FPGA target device.
    template<typename TDim, typename TIdx>
    class AccFpgaSyclXilinx final
        : public AccGenericSycl<TDim, TIdx>
        , public concepts::Implements<ConceptAcc, AccFpgaSyclXilinx<TDim, TIdx>>
    {
    public:
        using AccGenericSycl<TDim, TIdx>::AccGenericSycl;
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    //! The Xilinx FPGA SYCL accelerator name trait specialization.
    template<typename TDim, typename TIdx>
    struct GetAccName<experimental::AccFpgaSyclXilinx<TDim, TIdx>>
    {
        static auto getAccName() -> std::string
        {
            return "experimental::AccFpgaSyclXilinx<" + std::to_string(TDim::value) + ","
                   + core::demangled<TIdx> + ">";
        }
    };

    //! The Xilinx FPGA SYCL accelerator device type trait specialization.
    template<typename TDim, typename TIdx>
    struct DevType<experimental::AccFpgaSyclXilinx<TDim, TIdx>>
    {
        using type = experimental::DevFpgaSyclXilinx;
    };

    //! The Xilinx FPGA SYCL accelerator execution task type trait specialization.
    template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
    struct CreateTaskKernel<experimental::AccFpgaSyclXilinx<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
    {
        static auto createTaskKernel(TWorkDiv const& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
        {
            return experimental::TaskKernelFpgaSyclXilinx<TDim, TIdx, TKernelFnObj, TArgs...>{
                workDiv,
                kernelFnObj,
                std::forward<TArgs>(args)...};
        }
    };

    //! The Xilinx FPGA SYCL execution task platform type trait specialization.
    template<typename TDim, typename TIdx>
    struct PltfType<experimental::AccFpgaSyclXilinx<TDim, TIdx>>
    {
        using type = experimental::PltfFpgaSyclXilinx;
    };

    template<typename TDim, typename TIdx>
    struct AccToTag<alpaka::experimental::AccFpgaSyclXilinx<TDim, TIdx>>
    {
        using type = alpaka::TagFpgaSyclXilinx;
    };

    template<typename TDim, typename TIdx>
    struct TagToAcc<alpaka::TagFpgaSyclXilinx, TDim, TIdx>
    {
        using type = alpaka::experimental::AccFpgaSyclXilinx<TDim, TIdx>;
    };
} // namespace alpaka::trait

#endif
