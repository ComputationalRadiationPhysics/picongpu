/* Copyright 2013-2021 Heiko Burau, Rene Widera, Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace pmacc
{
    namespace allocator
    {
        template<typename Type, int T_dim>
        HDINLINE cursor::BufferCursor<Type, T_dim> DeviceMemAllocator<Type, T_dim>::allocate(
            const math::Size_t<T_dim>& size)
        {
#ifndef __CUDA_ARCH__
            Type* dataPointer;
            math::Size_t<T_dim - 1> pitch;
            cuplaPitchedPtr cuplaData;

            cuplaData.ptr = nullptr;
            cuplaData.pitch = 1;
            cuplaData.xsize = size[0] * sizeof(Type);
            cuplaData.ysize = 1;

            if(dim == 2u)
            {
                cuplaData.xsize = size[0] * sizeof(Type);
                cuplaData.ysize = size[1];
                if(size.productOfComponents())
                    CUDA_CHECK(cuplaMallocPitch(&cuplaData.ptr, &cuplaData.pitch, cuplaData.xsize, cuplaData.ysize));
                pitch[0] = cuplaData.pitch;
            }
            else if(dim == 3u)
            {
                cuplaExtent extent;
                extent.width = size[0] * sizeof(Type);
                extent.height = size[1];
                extent.depth = size[2];
                if(size.productOfComponents())
                    CUDA_CHECK(cuplaMalloc3D(&cuplaData, extent));
                pitch[0] = cuplaData.pitch;
                pitch[1] = cuplaData.pitch * size[1];
            }
            dataPointer = (Type*) cuplaData.ptr;

            return cursor::BufferCursor<Type, T_dim>(dataPointer, pitch);
#endif

#ifdef __CUDA_ARCH__
            Type* dataPointer = nullptr;
            math::Size_t<T_dim - 1> pitch;
            return cursor::BufferCursor<Type, T_dim>(dataPointer, pitch);
#endif
        }

        template<typename Type>
        HDINLINE cursor::BufferCursor<Type, 1> DeviceMemAllocator<Type, 1>::allocate(const math::Size_t<1>& size)
        {
#ifndef __CUDA_ARCH__
            Type* dataPointer = nullptr;

            if(size[0])
                CUDA_CHECK(cuplaMalloc((void**) &dataPointer, size[0] * sizeof(Type)));

            return cursor::BufferCursor<Type, 1>(dataPointer, math::Size_t<0>());
#endif

#ifdef __CUDA_ARCH__
            Type* dataPointer = nullptr;
            return cursor::BufferCursor<Type, 1>(dataPointer, math::Size_t<0>());
#endif
        }

        template<typename Type, int T_dim>
        template<typename TCursor>
        HDINLINE void DeviceMemAllocator<Type, T_dim>::deallocate(const TCursor& cursor)
        {
#ifndef __CUDA_ARCH__
            CUDA_CHECK(cuplaFree(cursor.getMarker()));
#endif
        }

        template<typename Type>
        template<typename TCursor>
        HDINLINE void DeviceMemAllocator<Type, 1>::deallocate(const TCursor& cursor)
        {
#ifndef __CUDA_ARCH__
            CUDA_CHECK(cuplaFree(cursor.getMarker()));
#endif
        }

    } // namespace allocator
} // namespace pmacc
