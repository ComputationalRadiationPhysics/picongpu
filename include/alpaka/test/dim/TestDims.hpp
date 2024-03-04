/* Copyright 2023 Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/meta/Filter.hpp"
#include "alpaka/meta/NonZero.hpp"

#include <tuple>

namespace alpaka::test
{
    //! A std::tuple holding dimensions.
    using TestDims = std::tuple<
        DimInt<0u>,
        DimInt<1u>,
        DimInt<2u>,
        DimInt<3u>
    // CUDA, HIP and SYCL accelerators do not support 4D buffers and 4D acceleration.
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !defined(ALPAKA_ACC_SYCL_ENABLED)
        ,
        DimInt<4u>
#endif
        >;

    //! A std::tuple holding non-zero dimensions.
    //!
    //! NonZeroTestDims = std::tuple<Dim1, Dim2, ... DimN>
    using NonZeroTestDims = meta::Filter<TestDims, meta::NonZero>;

} // namespace alpaka::test
