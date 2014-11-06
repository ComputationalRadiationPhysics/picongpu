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

#ifndef ALLOCATOR_HOSTMEMALLOCATOR_HPP
#define ALLOCATOR_HOSTMEMALLOCATOR_HPP

#include <stdint.h>
#include "math/vector/Size_t.hpp"
#include "cuSTL/cursor/BufferCursor.hpp"
#include "tag.h"

namespace PMacc
{
namespace allocator
{

template<typename Type, int _dim>
struct HostMemAllocator
{
    typedef Type type;
    static const int dim = _dim;
    typedef cursor::BufferCursor<type, dim> Cursor;
    typedef allocator::tag::host tag;

    HDINLINE
    static cursor::BufferCursor<type, dim> allocate(const math::Size_t<dim>& size);
    template<typename TCursor>
    HDINLINE
    static void deallocate(const TCursor& cursor);
};

template<typename Type>
struct HostMemAllocator<Type, 1>
{
    typedef Type type;
    static const int dim = 1;
    typedef cursor::BufferCursor<type, dim> Cursor;
    typedef allocator::tag::host tag;

    HDINLINE
    static cursor::BufferCursor<type, dim> allocate(const math::Size_t<dim>& size);
    template<typename TCursor>
    HDINLINE
    static void deallocate(const TCursor& cursor);
};

} // allocator
} // PMacc

#include "HostMemAllocator.tpp"

#endif // ALLOCATOR_HOSTMEMALLOCATOR_HPP
