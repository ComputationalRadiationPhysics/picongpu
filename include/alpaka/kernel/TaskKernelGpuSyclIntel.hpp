/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/kernel/TaskKernelGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    template<typename TDim, typename TIdx>
    class AccGpuSyclIntel;

    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    using TaskKernelGpuSyclIntel
        = TaskKernelGenericSycl<AccGpuSyclIntel<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>;

    namespace trait
    {
        //! \brief Specialisation of the class template FunctionAttributes
        //! \tparam TDev The device type.
        //! \tparam TDim The dimensionality of the accelerator device properties.
        //! \tparam TIdx The idx type of the accelerator device properties.
        //! \tparam TKernelFn Kernel function object type.
        //! \tparam TArgs Kernel function object argument types as a parameter pack.
        template<typename TDev, typename TDim, typename TIdx, typename TKernelFn, typename... TArgs>
        struct FunctionAttributes<AccGpuSyclIntel<TDim, TIdx>, TDev, KernelBundle<TKernelFn, TArgs...>>
        {
            //! \param dev The device instance
            //! \param kernelBundle Kernel bundeled with it's arguments. The function attributes of this kernel will be
            //! determined. Max threads per block is one of the attributes.
            //! \return KernelFunctionAttributes instance. The default version always returns an instance with zero
            //! fields. For CPU, the field of max threads allowed by kernel function for the block is 1.
            ALPAKA_FN_HOST static auto getFunctionAttributes(
                TDev const& dev,
                [[maybe_unused]] KernelBundle<TKernelFn, TArgs...> const& kernelBundle)
                -> alpaka::KernelFunctionAttributes
            {
                alpaka::KernelFunctionAttributes kernelFunctionAttributes;

                // set function properties for maxThreadsPerBlock to device properties
                auto const& props = alpaka::getAccDevProps<AccGpuSyclIntel<TDim, TIdx>>(dev);
                kernelFunctionAttributes.maxThreadsPerBlock = static_cast<int>(props.m_blockThreadCountMax);
                return kernelFunctionAttributes;
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
