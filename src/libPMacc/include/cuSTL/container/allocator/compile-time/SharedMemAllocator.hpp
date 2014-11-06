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

#ifndef ALLOCATOR_SHAREDMEMALLOCATOR_HPP
#define ALLOCATOR_SHAREDMEMALLOCATOR_HPP

#include "math/Vector.hpp"
#include "cuSTL/cursor/compile-time/BufferCursor.hpp"

namespace PMacc
{
namespace allocator
{
namespace CT
{
template<typename Type, typename Size, int dim = Size::dim, int uid = 0>
struct SharedMemAllocator;

template<typename Type, typename Size, int uid>
struct SharedMemAllocator<Type, Size, 1, uid>
{
    typedef Type type;
    typedef math::CT::UInt<> Pitch;
    static const int dim = 1;
    typedef cursor::CT::BufferCursor<type, math::CT::UInt<> > Cursor;

    __device__ static Cursor allocate()
    {
        __shared__ Type shMem[Size::x::value];
        __syncthreads(); /*wait that all shared memory is initialised*/
        return Cursor((Type*)shMem);
    }
};

template<typename Type, typename Size, int uid>
struct SharedMemAllocator<Type, Size, 2, uid>
{
    typedef Type type;
    typedef math::CT::UInt<sizeof(Type) * Size::x::value> Pitch;
    static const int dim = 2;
    typedef cursor::CT::BufferCursor<type, Pitch> Cursor;

    __device__ static Cursor allocate()
    {
        __shared__ Type shMem[Size::x::value][Size::y::value];
        __syncthreads(); /*wait that all shared memory is initialised*/
        return Cursor((Type*)shMem);
    }
};

template<typename Type, typename Size, int uid>
struct SharedMemAllocator<Type, Size, 3, uid>
{
    typedef Type type;
    typedef math::CT::UInt<sizeof(Type) * Size::x::value,
                             sizeof(Type) * Size::x::value * Size::y::value> Pitch;
    static const int dim = 3;
    typedef cursor::CT::BufferCursor<type, Pitch> Cursor;

    __device__ static Cursor allocate()
    {
        __shared__ Type shMem[Size::x::value][Size::y::value][Size::z::value];
        __syncthreads(); /*wait that all shared memory is initialised*/
        return Cursor((Type*)shMem);
    }
};

} // CT
} // allocator
} // PMacc

#endif // ALLOCATOR_SHAREDMEMALLOCATOR_HPP
