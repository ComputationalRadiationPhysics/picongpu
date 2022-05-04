/* Copyright 2022 Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Bernhard Manfred Gruber,
 * Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/queue/cuda_hip/QueueUniformCudaHipRt.hpp>

namespace alpaka
{
    //! The CUDA/HIP RT blocking queue.
    using QueueUniformCudaHipRtBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<true>;

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    using QueueCudaRtBlocking = QueueUniformCudaHipRtBlocking;
#    else
    using QueueHipRtBlocking = QueueUniformCudaHipRtBlocking;
#    endif

} // namespace alpaka

#endif
