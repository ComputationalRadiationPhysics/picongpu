/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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

#pragma once

namespace PMacc
{
namespace container
{
namespace CT
{

template<typename Type, typename _Size, typename Allocator, typename Copier, typename Assigner>
DINLINE CartBuffer<Type, _Size, Allocator, Copier, Assigner>::CartBuffer()
{
    this->dataPointer = Allocator::allocate().getMarker();
}

template<typename Type, typename _Size, typename Allocator, typename Copier, typename Assigner>
DINLINE
CartBuffer<Type, _Size, Allocator, Copier, Assigner>::Cursor
CartBuffer<Type, _Size, Allocator, Copier, Assigner>::origin() const
{
    return CartBuffer<Type, _Size, Allocator, Copier, Assigner>::Cursor(this->dataPointer);
}

template<typename Type, typename _Size, typename Allocator, typename Copier, typename Assigner>
DINLINE
CartBuffer<Type, _Size, Allocator, Copier, Assigner>::SafeCursor
CartBuffer<Type, _Size, Allocator, Copier, Assigner>::originSafe() const
{
    return CartBuffer<Type, _Size, Allocator, Copier, Assigner>::SafeCursor(this->origin());
}

} // CT
} // container
} // PMacc
