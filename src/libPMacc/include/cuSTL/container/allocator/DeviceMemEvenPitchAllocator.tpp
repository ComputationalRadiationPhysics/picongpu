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
DeviceMemEvenPitch<Type, _dim>::allocate(const math::Size_t<_dim>& size)
{
    Type* dataPointer;
    math::Size_t<_dim-1> pitch;

    CUDA_CHECK(cudaMalloc((void**)&dataPointer, sizeof(Type) * size.productOfComponents()));
    
    if (dim == 2u)
    {
        pitch[0] = sizeof(Type) * size[0];
    }
    else if (dim == 3u)
    {
        pitch[0] = sizeof(Type) * size[0];
        pitch[1] = pitch[0] * size[1];
    }
    
    return cursor::BufferCursor<Type, _dim>(dataPointer, pitch);
}

template<typename Type>
cursor::BufferCursor<Type, 1>
DeviceMemEvenPitch<Type, 1>::allocate(const math::Size_t<1>& size)
{
    Type* dataPointer;

    CUDA_CHECK(cudaMalloc((void**)&dataPointer, size[0] * sizeof(Type)));
    
    return cursor::BufferCursor<Type, 1>(dataPointer, math::Size_t<0>());
}

template<typename Type, int _dim>
template<typename TCursor>
void DeviceMemEvenPitch<Type, _dim>::deallocate(const TCursor& cursor)
{
    CUDA_CHECK(cudaFree(cursor.getMarker()));
}

template<typename Type>
template<typename TCursor>
void DeviceMemEvenPitch<Type, 1>::deallocate(const TCursor& cursor)
{
    CUDA_CHECK(cudaFree(cursor.getMarker()));
}

} // allocator
} // PMacc
