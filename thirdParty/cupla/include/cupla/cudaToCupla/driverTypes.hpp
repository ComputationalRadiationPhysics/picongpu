/* Copyright 2015-2020 Rene Widera, Maximilian Knespel, Matthias Werner,
 *                     Sergei Bastrakov
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


#pragma once

#include "cupla/datatypes/Array.hpp"
#include "cupla/device/SharedMemory.hpp"
#include "cupla/device_functions.hpp"

#define __syncthreads(...) ::cupla::syncThreads(acc)

#define cudaSuccess cuplaSuccess
#define cudaErrorMemoryAllocation cuplaErrorMemoryAllocation
#define cudaErrorInitializationError cuplaErrorInitializationError
#define cudaErrorNotReady cuplaErrorNotReady
#define cudaErrorDeviceAlreadyInUse cuplaErrorDeviceAlreadyInUse
#define cudaErrorInvalidDevice cuplaErrorInvalidDevice

#define cudaError_t cuplaError_t
#define cudaError cuplaError

#define cudaEvent_t cuplaEvent_t

#define cudaStream_t cuplaStream_t

#define dim3 cupla::dim3
#define cudaExtent cuplaExtent
#define cudaPos cuplaPos
#define cudaArray cuplaArray;
#define cudaPitchedPtr cuplaPitchedPtr

#define cudaMemcpy3DParms cuplaMemcpy3DParms

#ifdef cudaEventBlockingSync
#undef cudaEventBlockingSync
#endif
/* cudaEventBlockingSync is a define in CUDA, hence we must remove
 * the old definition with the cupla enum
 */
#define cudaEventBlockingSync cuplaEventBlockingSync

#ifdef cudaEventDisableTiming
#undef cudaEventDisableTiming
#endif
/* cudaEventDisableTiming is a define in CUDA therefore we must remove
 * the old definition with the cupla enum
 */
#define cudaEventDisableTiming cuplaEventDisableTiming

#define cudaMemcpyKind cuplaMemcpyKind
#define cudaMemcpyHostToDevice cuplaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost cuplaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice cuplaMemcpyDeviceToDevice
#define cudaMemcpyHostToHost cuplaMemcpyHostToHost

// index renaming
#define blockIdx cupla::blockIdx(acc)
#define threadIdx cupla::threadIdx(acc)
#define gridDim cupla::gridDim(acc)
#define blockDim cupla::blockDim(acc)
#define elemDim cupla::threadDim(acc)

/** Atomic functions
 *
 * Compared to their CUDA counterparts, these functions take an additional last
 * parameter to denote atomicity (synchronization) level. This parameter is
 * of type ::alpaka::hierarchy::{Grids|Blocks|Threads}. Grids corresponds
 * to atomicity between different kernels, Blocks - to different blocks
 * in the same kernel, Threads - to threads of the same block.
 *
 * @{
 */
#define atomicAdd(...) cupla::atomicAdd(acc, __VA_ARGS__)
#define atomicSub(...) cupla::atomicSub(acc, __VA_ARGS__)
#define atomicMin(...) cupla::atomicMin(acc, __VA_ARGS__)
#define atomicMax(...) cupla::atomicMax(acc, __VA_ARGS__)
#define atomicInc(...) cupla::atomicInc(acc, __VA_ARGS__)
#define atomicDec(...) cupla::atomicDec(acc, __VA_ARGS__)
#define atomicExch(...) cupla::atomicExch(acc, __VA_ARGS__)
#define atomicCAS(...) cupla::atomicCAS(acc, __VA_ARGS__)
#define atomicAnd(...) cupla::atomicAnd(acc, __VA_ARGS__)
#define atomicXor(...) cupla::atomicXor(acc, __VA_ARGS__)
#define atomicOr(...) cupla::atomicOr(acc, __VA_ARGS__)
/** @} */

#define uint3 ::cupla::uint3

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)

//! Simplistic stub for CUDA float3
struct float3
{
    float x, y, z;
};

//! Factory function for float3
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float3 make_float3(float x, float y, float z)
{
    return float3{x, y, z};
}

//! Simplistic stub for CUDA int3
struct int3
{
    int x, y, z;
};

//! Factory function for int3
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int3 make_int3(int x, int y, int z)
{
    return int3{x, y, z};
}

#endif

// recast functions
namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

    template< typename A, typename B >
    ALPAKA_FN_HOST_ACC
    B A_as_B( A const & x )
    {
        static_assert( sizeof(A) == sizeof(B), "reinterpretation assumes data types of same size!" );
        return reinterpret_cast< B const & >( x );
    }

} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla

#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
#   define __int_as_float(...) cupla::A_as_B< int, float >( __VA_ARGS__ )
#   define __float_as_int(...) cupla::A_as_B< float, int >( __VA_ARGS__ )
#   define __longlong_as_double(...) cupla::A_as_B< long long, double >( __VA_ARGS__ )
#   define __double_as_longlong(...) cupla::A_as_B< double, long long >( __VA_ARGS__ )
#endif
