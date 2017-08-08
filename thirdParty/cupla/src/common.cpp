/**
 * Copyright 2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#include "cupla_runtime.hpp"
#include "cupla/manager/Memory.hpp"
#include "cupla/manager/Device.hpp"
#include "cupla/manager/Stream.hpp"
#include "cupla/manager/Event.hpp"
#include "cupla/api/common.hpp"


const char *
cuplaGetErrorString(cuplaError_t)
{
    return "cuplaGetErrorString is currently not supported\n";
}

cuplaError_t
cuplaGetLastError()
{
#if (ALPAKA_ACC_GPU_CUDA_ENABLED == 1)
    // reset the last cuda error
    cudaGetLastError();
#endif
    return cuplaSuccess;
}

cuplaError_t
cuplaPeekAtLastError()
{
    return cuplaSuccess;
}
