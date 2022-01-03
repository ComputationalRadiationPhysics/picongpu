/* Copyright 2013-2022 Heiko Burau, Rene Widera, Alexander Grund
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
        HINLINE cursor::BufferCursor<Type, T_dim> HostMemAllocator<Type, T_dim>::allocate(
            const math::Size_t<T_dim>& size)
        {
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
        }

        template<typename Type>
        HINLINE cursor::BufferCursor<Type, 1> HostMemAllocator<Type, 1>::allocate(const math::Size_t<1>& size)
        {
            Type* dataPointer = nullptr;
            math::Size_t<0> pitch;

            if(size.productOfComponents())
                CUDA_CHECK(cuplaMallocHost((void**) &dataPointer, sizeof(Type) * size.productOfComponents()));

            return cursor::BufferCursor<Type, 1>(dataPointer, pitch);
        }

        template<typename Type, int T_dim>
        template<typename TDataPtr>
        HINLINE void HostMemAllocator<Type, T_dim>::deallocate(const TDataPtr* ptr)
        {
            CUDA_CHECK(cuplaFreeHost(const_cast<TDataPtr*>(ptr)));
        }

        template<typename Type>
        template<typename TDataPtr>
        HINLINE void HostMemAllocator<Type, 1>::deallocate(const TDataPtr* ptr)
        {
            CUDA_CHECK(cuplaFreeHost(const_cast<TDataPtr*>(ptr)));
        }

    } // namespace allocator
} // namespace pmacc
