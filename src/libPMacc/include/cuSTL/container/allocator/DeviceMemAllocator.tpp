/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

namespace PMacc
{
namespace allocator
{

template<typename Type, int _dim>
cursor::BufferCursor<Type, _dim>
DeviceMemAllocator<Type, _dim>::allocate(const math::Size_t<_dim>& size)
{
#ifndef __CUDA_ARCH__
    Type* dataPointer;
    math::Size_t<_dim-1> pitch;
    cudaPitchedPtr cudaData;

    cudaData.ptr = NULL;
    cudaData.pitch = 1;
    cudaData.xsize = size.x();
    cudaData.ysize = 1;

    if (dim == 2u)
    {
        cudaData.xsize = size[0];
        cudaData.ysize = size[1];
        CUDA_CHECK_NO_EXCEP(cudaMallocPitch(&cudaData.ptr, &cudaData.pitch, cudaData.xsize * sizeof (Type), cudaData.ysize));
        pitch[0] = cudaData.pitch;
    }
    else if (dim == 3u)
    {
        cudaExtent extent;
        extent.width = size[0] * sizeof (Type);
        extent.height = size[1];
        extent.depth = size[2];
        CUDA_CHECK_NO_EXCEP(cudaMalloc3D(&cudaData, extent));
        pitch[0] = cudaData.pitch;
        pitch[1] = cudaData.pitch * size[1];
    }
    dataPointer = (Type*)cudaData.ptr;

    return cursor::BufferCursor<Type, _dim>(dataPointer, pitch);
#endif

#ifdef __CUDA_ARCH__
    Type* dataPointer = 0;
    math::Size_t<_dim-1> pitch;
    return cursor::BufferCursor<Type, _dim>(dataPointer, pitch);
#endif
}

template<typename Type>
cursor::BufferCursor<Type, 1>
DeviceMemAllocator<Type, 1>::allocate(const math::Size_t<1>& size)
{
#ifndef __CUDA_ARCH__
    Type* dataPointer;

    CUDA_CHECK_NO_EXCEP(cudaMalloc((void**)&dataPointer, size[0] * sizeof(Type)));

    return cursor::BufferCursor<Type, 1>(dataPointer, math::Size_t<0>());
#endif

#ifdef __CUDA_ARCH__
    Type* dataPointer = 0;
    return cursor::BufferCursor<Type, 1>(dataPointer, math::Size_t<0>());
#endif
}

template<typename Type, int _dim>
template<typename TCursor>
void DeviceMemAllocator<Type, _dim>::deallocate(const TCursor& cursor)
{
#ifndef __CUDA_ARCH__
    CUDA_CHECK_NO_EXCEP(cudaFree(cursor.getMarker()));
#endif
}

template<typename Type>
template<typename TCursor>
void DeviceMemAllocator<Type, 1>::deallocate(const TCursor& cursor)
{
#ifndef __CUDA_ARCH__
    CUDA_CHECK_NO_EXCEP(cudaFree(cursor.getMarker()));
#endif
}

} // allocator
} // PMacc
