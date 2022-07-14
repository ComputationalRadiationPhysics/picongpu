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


#pragma once

#include "cupla/c/datatypes/cuplaExtent.hpp"
#include "cupla/c/datatypes/cuplaMemcpy3DParms.hpp"
#include "cupla/c/datatypes/cuplaPitchedPtr.hpp"
#include "cupla/c/datatypes/cuplaPos.hpp"
#include "cupla/datatypes/dim3.hpp"
#include "cupla/datatypes/uint.hpp"
#include "cupla/namespace.hpp"
#include "cupla/types.hpp"
#include "cupla_driver_types.hpp"

#include <alpaka/alpaka.hpp>

inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
    /** allocate device memory
     *
     * @param[out] ptrptr Address to store the address of the allocated chunk.
     *                    Will be set to nullptr if allocation is failing.
     * @param size Aize to allocate (in byte).
     *
     * @return cuplaSuccess if allocation succeed, else cuplaErrorMemoryAllocation.
     */
    cuplaError_t cuplaMalloc(void** ptrptr, size_t size);

    /** allocate pinned host memory
     *
     * @param[out] ptrptr Address to store the address of the allocated chunk.
     *                    Will be set to nullptr if allocation is failing.
     * @param size Aize to allocate (in byte).
     *
     * @return cuplaSuccess if allocation succeed, else cuplaErrorMemoryAllocation.
     */
    cuplaError_t cuplaMallocHost(void** ptrptr, size_t size);

    /** allocate 2D memory
     *
     * @param[out] devPtr Address to store the address of the allocated memory chunk.
     *                    Will be set to nullptr if allocation is failing.
     * @param[out] pitch address to store the number of bytes of pitch
     *                   Will be set to zero if allocation is failing.
     * @param width with of the memory chunk (in byte)
     * @param height number of rows
     *
     * @return cuplaSuccess if allocation succeed, else cuplaErrorMemoryAllocation
     */
    cuplaError_t cuplaMallocPitch(void** devPtr, size_t* pitch, size_t const width, size_t const height);

    /** allocate memory
     *
     * @param[out] pitchedDevPtr address pitch pointer data
     *                           Pitch pointer pointer will be set to nullptr if allocation is failing.
     * @param extent extent of the memory to allocate
     *
     * @return cuplaSuccess if allocation succeed, else cuplaErrorMemoryAllocation
     */
    cuplaError_t cuplaMalloc3D(cuplaPitchedPtr* pitchedDevPtr, cuplaExtent const extent);


    cuplaExtent make_cuplaExtent(size_t const w, size_t const h, size_t const d);

    cuplaPos make_cuplaPos(size_t const x, size_t const y, size_t const z);

    cuplaPitchedPtr make_cuplaPitchedPtr(void* const d, size_t const p, size_t const xsz, size_t const ysz);

    /** free device memory allocated with cuplaMalloc, cuplaMallocPitch, or cuplaMalloc3D
     *
     * @param ptr Pointer to free, nullptr is allowed.
     */
    cuplaError_t cuplaFree(void* ptr);

    /** free host memory allocated with cuplaMallocHost
     *
     * @param ptr Pointer to free, nullptr is allowed.
     */
    cuplaError_t cuplaFreeHost(void* ptr);

    cuplaError_t cuplaMemcpy(void* dst, const void* src, size_t count, enum cuplaMemcpyKind kind);

    cuplaError_t cuplaMemcpyAsync(
        void* dst,
        const void* src,
        size_t count,
        enum cuplaMemcpyKind kind,
        cuplaStream_t stream = 0);

    cuplaError_t cuplaMemsetAsync(void* devPtr, int value, size_t count, cuplaStream_t stream = 0);

    cuplaError_t cuplaMemset(void* devPtr, int value, size_t count);

    cuplaError_t cuplaMemcpy2D(
        void* dst,
        size_t const dPitch,
        void const* const src,
        size_t const spitch,
        size_t const width,
        size_t const height,
        enum cuplaMemcpyKind kind);

    cuplaError_t cuplaMemcpy2DAsync(
        void* dst,
        size_t const dPitch,
        void const* const src,
        size_t const spitch,
        size_t const width,
        size_t const height,
        enum cuplaMemcpyKind kind,
        cuplaStream_t const stream = 0);

    cuplaError_t cuplaMemcpy3DAsync(const cuplaMemcpy3DParms* const p, cuplaStream_t stream = 0);

    cuplaError_t cuplaMemcpy3D(const cuplaMemcpy3DParms* const p);

} // namespace CUPLA_ACCELERATOR_NAMESPACE
