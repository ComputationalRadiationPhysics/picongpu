/* Copyright 2015-2016 Rene Widera
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

#define cudaGetDeviceCount(...) cuplaGetDeviceCount(__VA_ARGS__)

#define cudaMalloc(...) cuplaMalloc(__VA_ARGS__)
#define cudaMallocPitch(...) cuplaMallocPitch(__VA_ARGS__)
#define cudaMalloc3D(...) cuplaMalloc3D(__VA_ARGS__)
#define cudaMallocHost(...) cuplaMallocHost(__VA_ARGS__)

#define cudaGetErrorString(...) cuplaGetErrorString(__VA_ARGS__)

#define cudaFree(...) cuplaFree(__VA_ARGS__)
#define cudaFreeHost(...) cuplaFreeHost(__VA_ARGS__)

#define cudaSetDevice(...) cuplaSetDevice(__VA_ARGS__)
#define cudaGetDevice(...) cuplaGetDevice(__VA_ARGS__)

#define cudaEventCreate(...) cuplaEventCreate(__VA_ARGS__)
#define cudaEventCreateWithFlags(...) cuplaEventCreateWithFlags(__VA_ARGS__)
#define cudaEventDestroy(...) cuplaEventDestroy(__VA_ARGS__)

#define cudaStreamCreate(...) cuplaStreamCreate(__VA_ARGS__)
#define cudaStreamDestroy(...) cuplaStreamDestroy(__VA_ARGS__)
#define cudaStreamSynchronize(...) cuplaStreamSynchronize(__VA_ARGS__)
#define cudaStreamWaitEvent(...) cuplaStreamWaitEvent(__VA_ARGS__)
#define cudaStreamQuery(...) cuplaStreamQuery(__VA_ARGS__)

#define cudaEventRecord(...) cuplaEventRecord(__VA_ARGS__)

#define cudaEventElapsedTime(...) cuplaEventElapsedTime(__VA_ARGS__)

#define cudaEventSynchronize(...) cuplaEventSynchronize(__VA_ARGS__)

#define cudaMemcpy(...) cuplaMemcpy(__VA_ARGS__)
#define cudaMemcpyAsync(...) cuplaMemcpyAsync(__VA_ARGS__)

#define cudaDeviceReset(...) cuplaDeviceReset(__VA_ARGS__)

#define cudaDeviceSynchronize(...) cuplaDeviceSynchronize(__VA_ARGS__)

#define cudaPeekAtLastError(...) cuplaPeekAtLastError(__VA_ARGS__)
#define cudaGetLastError(...) cuplaGetLastError(__VA_ARGS__)

#define cudaMemset(...) cuplaMemset(__VA_ARGS__)
#define cudaMemsetAsync(...) cuplaMemsetAsync(__VA_ARGS__)
#define cudaMemcpy2D(...) cuplaMemcpy2D(__VA_ARGS__)
#define cudaMemcpy2DAsync(...) cuplaMemcpy2DAsync(__VA_ARGS__)
#define cudaMemcpy3DAsync(...) cuplaMemcpy3DAsync(__VA_ARGS__)
#define cudaMemcpy3D(...) cuplaMemcpy3D(__VA_ARGS__)

#define cudaEventQuery(...) cuplaEventQuery(__VA_ARGS__)

#define cudaMemGetInfo(...) cuplaMemGetInfo(__VA_ARGS__)

#define make_cudaExtent(...) make_cuplaExtent(__VA_ARGS__)
#define make_cudaPos(...) make_cuplaPos(__VA_ARGS__)

#define make_cudaPitchedPtr(...) make_cuplaPitchedPtr(__VA_ARGS__)

/** define math intrinsics
 *
 * to avoid negative performance impact intrinsic function redefinitions
 * are disabled in CUDA
 */
#if !defined(__CUDA_ARCH__)
#define __fdividef(a,b) ((a)/(b))
#define __expf(a) alpaka::math::exp(acc,a)
#define __logf(a) alpaka::math::log(acc,a)
#endif
