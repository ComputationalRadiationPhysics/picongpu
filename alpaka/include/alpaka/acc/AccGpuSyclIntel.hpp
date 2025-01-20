/* Copyright 2024 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Sycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

namespace alpaka
{
    //! The Intel GPU SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Intel GPU target device.
    template<typename TDim, typename TIdx>
    using AccGpuSyclIntel = AccGenericSycl<TagGpuSyclIntel, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccGpuSyclIntel<TDim, TIdx>>
        {
            using type = alpaka::TagGpuSyclIntel;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagGpuSyclIntel, TDim, TIdx>
        {
            using type = alpaka::AccGpuSyclIntel<TDim, TIdx>;
        };
    } // namespace trait

} // namespace alpaka

#endif
