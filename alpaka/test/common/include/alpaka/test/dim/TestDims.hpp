/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dim/DimIntegralConst.hpp>

#include <tuple>

namespace alpaka
{
    namespace test
    {
        //#############################################################################
        //! A std::tuple holding dimensions.
        using TestDims = std::tuple<
            alpaka::DimInt<1u>,
            alpaka::DimInt<2u>,
            alpaka::DimInt<3u>
        // The CUDA & HIP accelerators do not currently support 4D buffers and 4D acceleration.
#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA)
#    if !(defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
            ,
            alpaka::DimInt<4u>
#    endif
#endif
            >;
    } // namespace test
} // namespace alpaka
