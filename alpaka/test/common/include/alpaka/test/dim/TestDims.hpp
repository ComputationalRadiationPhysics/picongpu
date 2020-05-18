/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dim/DimIntegralConst.hpp>

#include <tuple>

// When compiling the tests with CUDA enabled (nvcc or native clang) on the CI infrastructure
// we have to dramatically reduce the number of tested combinations.
// Else the log length would be exceeded.
#if defined(ALPAKA_CI)
  #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA \
   || defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP
    #define ALPAKA_CUDA_CI
  #endif
#endif

namespace alpaka
{
    namespace test
    {
        namespace dim
        {
            //#############################################################################
            //! A std::tuple holding dimensions.
            using TestDims =
                std::tuple<
                    alpaka::dim::DimInt<1u>
#if !defined(ALPAKA_CUDA_CI)
                    ,alpaka::dim::DimInt<2u>
#endif
                    ,alpaka::dim::DimInt<3u>
                    // The CUDA & HIP accelerators do not currently support 4D buffers and 4D acceleration.
#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA)
  #if !(defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                    ,alpaka::dim::DimInt<4u>
  #endif
#endif
                >;
        }
    }
}
