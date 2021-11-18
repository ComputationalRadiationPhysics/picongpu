/* Copyright 2019-2021 Benjamin Worpitz, René Widera, Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/UniformCudaHip.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#    include <cuda_runtime.h>
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <hip/math_functions.h>
#endif
