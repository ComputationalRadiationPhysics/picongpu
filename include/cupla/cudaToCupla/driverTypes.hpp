/* Copyright 2015-2016 Rene Widera, Maximilian Knespel
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

#define __syncthreads(...) ::alpaka::block::sync::syncBlockThreads(acc)

#define cudaSuccess cuplaSuccess
#define cudaErrorMemoryAllocation cuplaErrorMemoryAllocation
#define cudaErrorInitializationError cuplaErrorInitializationError
#define cudaErrorNotReady cuplaErrorNotReady
#define cudaErrorDeviceAlreadyInUse cuplaErrorDeviceAlreadyInUse
#define cuplaErrorInvalidDevice cuplaErrorInvalidDevice

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

#define sharedMem(ppName, ...)                                                 \
  __VA_ARGS__ &ppName =                                                        \
      ::alpaka::block::shared::st::allocVar<__VA_ARGS__, __COUNTER__>(acc)

#define sharedMemExtern(ppName, ...)                                           \
    __VA_ARGS__* ppName =                                                      \
        ::alpaka::block::shared::dyn::getMem<__VA_ARGS__>(acc)

#define cudaMemcpyKind cuplaMemcpyKind
#define cudaMemcpyHostToDevice cuplaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost cuplaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice cuplaMemcpyDeviceToDevice
#define cudaMemcpyHostToHost cuplaMemcpyHostToHost

// index renaming
#define blockIdx                                                               \
  static_cast<uint3>(                                                \
      ::alpaka::idx::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc))
#define threadIdx                                                              \
  static_cast<uint3>(                                                \
      ::alpaka::idx::getIdx<::alpaka::Block, ::alpaka::Threads>(acc))

#define gridDim                                                                \
  static_cast<uint3>(                                                \
      ::alpaka::workdiv::getWorkDiv<::alpaka::Grid, ::alpaka::Blocks>(acc))
#define blockDim                                                               \
  static_cast<uint3>(                                                \
      ::alpaka::workdiv::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(acc))
#define elemDim                                                               \
  static_cast<uint3>(                                                \
      ::alpaka::workdiv::getWorkDiv<::alpaka::Thread, ::alpaka::Elems>(acc))

// atomic functions
#define atomicAdd(...) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Add>(acc, __VA_ARGS__)
#define atomicSub(...) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Sub>(acc, __VA_ARGS__)
#define atomicMin(...) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Min>(acc, __VA_ARGS__)
#define atomicMax(...) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Max>(acc, __VA_ARGS__)
#define atomicInc(...) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Inc>(acc, __VA_ARGS__)
#define atomicDec(...) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Dec>(acc, __VA_ARGS__)
#define atomicExch(...) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Exch>(acc, __VA_ARGS__)
#define atomicCAS(...) ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Cas>(acc, __VA_ARGS__)

#define uint3 ::cupla::uint3

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
