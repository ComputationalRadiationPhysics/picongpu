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
        HDINLINE cursor::BufferCursor<Type, T_dim> HostMemAllocator<Type, T_dim>::allocate(
            const math::Size_t<T_dim>& size)
        {
#ifndef __CUDA_ARCH__
            Type* dataPointer = nullptr;
            math::Size_t<T_dim - 1> pitch;

            if(size.productOfComponents())
                CUDA_CHECK(cuplaMallocHost((void**) &dataPointer, sizeof(Type) * size.productOfComponents()));
            if(dim == 2u)
            {
                pitch[0] = size[0] * sizeof(Type);
            }
            else if(dim == 3u)
            {
                pitch[0] = size[0] * sizeof(Type);
                pitch[1] = pitch[0] * size[1];
            }

            return cursor::BufferCursor<Type, T_dim>(dataPointer, pitch);
#endif

#ifdef __CUDA_ARCH__
            Type* dataPointer = nullptr;
            math::Size_t<T_dim - 1> pitch;
            return cursor::BufferCursor<Type, T_dim>(dataPointer, pitch);
#endif
        }

        template<typename Type>
        HDINLINE cursor::BufferCursor<Type, 1> HostMemAllocator<Type, 1>::allocate(const math::Size_t<1>& size)
        {
#ifndef __CUDA_ARCH__
            Type* dataPointer = nullptr;
            math::Size_t<0> pitch;

            if(size.productOfComponents())
                CUDA_CHECK(cuplaMallocHost((void**) &dataPointer, sizeof(Type) * size.productOfComponents()));

            return cursor::BufferCursor<Type, 1>(dataPointer, pitch);
#endif

#ifdef __CUDA_ARCH__
            Type* dataPointer = nullptr;
            math::Size_t<0> pitch;
            return cursor::BufferCursor<Type, 1>(dataPointer, pitch);
#endif
        }

        template<typename Type, int T_dim>
        template<typename TCursor>
        HDINLINE void HostMemAllocator<Type, T_dim>::deallocate(const TCursor& cursor)
        {
#ifndef __CUDA_ARCH__
            CUDA_CHECK(cuplaFreeHost(cursor.getMarker()));
#endif
        }

        template<typename Type>
        template<typename TCursor>
        HDINLINE void HostMemAllocator<Type, 1>::deallocate(const TCursor& cursor)
        {
#ifndef __CUDA_ARCH__
            CUDA_CHECK(cuplaFreeHost(cursor.getMarker()));
#endif
        }

    } // namespace allocator
} // namespace pmacc
