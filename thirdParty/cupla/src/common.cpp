/* Copyright 2016 Rene Widera
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


#include "cupla/namespace.hpp"
#include "cupla_runtime.hpp"
#include "cupla/manager/Memory.hpp"
#include "cupla/manager/Device.hpp"
#include "cupla/manager/Stream.hpp"
#include "cupla/manager/Event.hpp"
#include "cupla/api/common.hpp"

inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

CUPLA_HEADER_ONLY_FUNC_SPEC
const char *
cuplaGetErrorName(cuplaError_t e)
{
    return CuplaErrorCode::message_cstr(e);
}

CUPLA_HEADER_ONLY_FUNC_SPEC
const char *
cuplaGetErrorString(cuplaError_t e)
{
    return CuplaErrorCode::message_cstr(e);
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaGetLastError()
{
#if( ALPAKA_ACC_GPU_CUDA_ENABLED == 1 )
    // reset the last cuda error
    return (cuplaError_t)cudaGetLastError();
#elif( ALPAKA_ACC_GPU_HIP_ENABLED == 1 )
    return (cuplaError_t)hipGetLastError();
#else
    return cuplaSuccess;
#endif
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaPeekAtLastError()
{
#if( ALPAKA_ACC_GPU_CUDA_ENABLED == 1 )
    return (cuplaError_t)cudaPeekAtLastError();
#elif( ALPAKA_ACC_GPU_HIP_ENABLED == 1 )
    return (cuplaError_t)hipPeekAtLastError();
#else
    return cuplaSuccess;
#endif
}

} //namespace CUPLA_ACCELERATOR_NAMESPACE
