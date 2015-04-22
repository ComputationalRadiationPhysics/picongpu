/**
 * Copyright 2014  Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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

#pragma once

#include "types.h"


namespace PMacc
{

/** wrapper for native C pointer
 *
 * @tparam T_Type type of the pointed object
 */
template <class T_Type>
class Pointer
{
public:

    typedef T_Type type;
    typedef type* PtrType;

    /** default constructor
     *
     * the default pointer points to invalid memory
     */
    HDINLINE Pointer() : ptr(NULL)
    {
    }

    HDINLINE Pointer(PtrType const ptrIn) : ptr(ptrIn)
    {
    }

    HDINLINE Pointer(const Pointer<type>& other) : ptr(other.ptr)
    {
    }

    /** dereference the pointer*/
    HDINLINE type& operator*()
    {
        return *ptr;
    }

    /** access member*/
    HDINLINE PtrType operator->()
    {
        return ptr;
    }

    /** compare if two pointers point to the same memory address*/
    HDINLINE bool operator==(const Pointer<type>& other) const
    {
        return ptr == other.ptr;
    }

    /** check if the memory address of two pointers are different*/
    HDINLINE bool operator!=(const Pointer<type>& other) const
    {
        return ptr != other.ptr;
    }

    /** check if the memory pointed to has a valid address
     * @return false if memory adress is NULL else true
     */
    HDINLINE bool isValid() const
    {
        return ptr != NULL;
    }

    PMACC_ALIGN(ptr, PtrType);
};

} //namespace PMacc
