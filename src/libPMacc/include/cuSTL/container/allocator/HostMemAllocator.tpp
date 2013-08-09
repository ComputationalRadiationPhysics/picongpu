/**
 * Copyright 2013 Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
HostMemAllocator<Type, _dim>::allocate(const math::Size_t<_dim>& size)
{
#ifndef __CUDA_ARCH__
    Type* dataPointer;
    math::Size_t<_dim-1> pitch;

    CUDA_CHECK_NO_EXCEP(cudaMallocHost((void**)&dataPointer, sizeof(Type) * size.volume()));
    if(dim == 2u)
    {
        pitch[0] = size[0] * sizeof(Type);
    }
    else if(dim == 3u)
    {
        pitch[0] = size[0] * sizeof(Type);
        pitch[1] = pitch[0] * size[1];
    }
    
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
HostMemAllocator<Type, 1>::allocate(const math::Size_t<1>& size)
{
#ifndef __CUDA_ARCH__
    Type* dataPointer;
    math::Size_t<0> pitch;

    CUDA_CHECK_NO_EXCEP(cudaMallocHost((void**)&dataPointer, sizeof(Type) * size.volume()));
    
    return cursor::BufferCursor<Type, 1>(dataPointer, pitch);
#endif

#ifdef __CUDA_ARCH__
    Type* dataPointer = 0;
    math::Size_t<0> pitch;
    return cursor::BufferCursor<Type, 1>(dataPointer, pitch);
#endif
}

template<typename Type, int _dim>
template<typename TCursor>
void HostMemAllocator<Type, _dim>::deallocate(const TCursor& cursor)
{
#ifndef __CUDA_ARCH__
    CUDA_CHECK_NO_EXCEP(cudaFreeHost(cursor.getMarker()));
#endif
}

template<typename Type>
template<typename TCursor>
void HostMemAllocator<Type, 1>::deallocate(const TCursor& cursor)
{
#ifndef __CUDA_ARCH__
    CUDA_CHECK_NO_EXCEP(cudaFreeHost(cursor.getMarker()));
#endif
}

} // allocator
} // PMacc
