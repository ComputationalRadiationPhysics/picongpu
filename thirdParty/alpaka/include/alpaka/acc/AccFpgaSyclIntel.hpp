/* Copyright 2024 Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGenericSycl.hpp"
#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Sycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

namespace alpaka
{
    //! The Intel FPGA SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on a oneAPI-capable Intel FPGA target device.
    template<typename TDim, typename TIdx>
    using AccFpgaSyclIntel = AccGenericSycl<TagFpgaSyclIntel, TDim, TIdx>;

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct AccToTag<alpaka::AccFpgaSyclIntel<TDim, TIdx>>
        {
            using type = alpaka::TagFpgaSyclIntel;
        };

        template<typename TDim, typename TIdx>
        struct TagToAcc<alpaka::TagFpgaSyclIntel, TDim, TIdx>
        {
            using type = alpaka::AccFpgaSyclIntel<TDim, TIdx>;
        };
    } // namespace trait

} // namespace alpaka

#endif
