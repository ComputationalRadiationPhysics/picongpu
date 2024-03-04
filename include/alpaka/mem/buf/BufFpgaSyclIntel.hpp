/* Copyright 2023 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevFpgaSyclIntel.hpp"
#include "alpaka/mem/buf/BufGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufFpgaSyclIntel = BufGenericSycl<TElem, TDim, TIdx, DevFpgaSyclIntel>;
} // namespace alpaka

#endif
