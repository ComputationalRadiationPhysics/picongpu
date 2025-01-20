/* Copyright 2024 Jan Stephan, Luca Ferragina, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Sycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)

namespace alpaka
{
    //! The CPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable CPU target device.
    template<typename TDim, typename TIdx>
    using AccCpuSycl = AccGenericSycl<TagCpuSycl, TDim, TIdx>;

    namespace trait
    {
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
    } // namespace trait

} // namespace alpaka

#endif
