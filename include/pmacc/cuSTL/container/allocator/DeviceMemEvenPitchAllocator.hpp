/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include "tag.hpp"
#include "pmacc/cuSTL/cursor/BufferCursor.hpp"

#include "pmacc/math/vector/Size_t.hpp"


namespace pmacc
{
    namespace allocator
    {
        template<typename Type, int T_dim>
        struct DeviceMemEvenPitch
        {
            typedef Type type;
            static constexpr int dim = T_dim;
            typedef cursor::BufferCursor<type, dim> Cursor;
            typedef allocator::tag::device tag;

            static cursor::BufferCursor<type, T_dim> allocate(const math::Size_t<T_dim>& size);
            template<typename TCursor>
            static void deallocate(const TCursor& cursor);
        };

        template<typename Type>
        struct DeviceMemEvenPitch<Type, 1>
        {
            typedef Type type;
            static constexpr int dim = 1;
            typedef cursor::BufferCursor<type, 1> Cursor;
            typedef allocator::tag::device tag;

            static cursor::BufferCursor<type, 1> allocate(const math::Size_t<1>& size);
            template<typename TCursor>
            static void deallocate(const TCursor& cursor);
        };

    } // namespace allocator
} // namespace pmacc

#include "DeviceMemEvenPitchAllocator.tpp"
