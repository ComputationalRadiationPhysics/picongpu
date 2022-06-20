/* Copyright 2022 Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/mem/buf/BufUniformCudaHipRt.hpp>

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufHipRt = BufUniformCudaHipRt<ApiHipRt, TElem, TDim, TIdx>;
}

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
