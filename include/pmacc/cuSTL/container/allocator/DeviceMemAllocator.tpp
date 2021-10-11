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
        HINLINE cursor::BufferCursor<Type, T_dim> DeviceMemAllocator<Type, T_dim>::allocate(
            const math::Size_t<T_dim>& size)
        {
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
        }

        template<typename Type>
        HINLINE cursor::BufferCursor<Type, 1> DeviceMemAllocator<Type, 1>::allocate(const math::Size_t<1>& size)
        {
            Type* dataPointer = nullptr;

            if(size[0])
                CUDA_CHECK(cuplaMalloc((void**) &dataPointer, size[0] * sizeof(Type)));

            return cursor::BufferCursor<Type, 1>(dataPointer, math::Size_t<0>());
        }

        template<typename Type, int T_dim>
        template<typename TDataPtr>
        HINLINE void DeviceMemAllocator<Type, T_dim>::deallocate(const TDataPtr* ptr)
        {
            CUDA_CHECK(cuplaFree(const_cast<TDataPtr*>(ptr)));
        }

        template<typename Type>
        template<typename TDataPtr>
        HINLINE void DeviceMemAllocator<Type, 1>::deallocate(const TDataPtr* ptr)
        {
            CUDA_CHECK(cuplaFree(const_cast<TDataPtr*>(ptr)));
        }

    } // namespace allocator
} // namespace pmacc
