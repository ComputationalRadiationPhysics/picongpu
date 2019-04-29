/* Copyright 2014-2019  Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
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

#include "pmacc/types.hpp"
#include "pmacc/traits/GetEmptyDefaultConstructibleType.hpp"
#include "pmacc/expressions/DoNothing.hpp"
#include "pmacc/expressions/SetToNull.hpp"


namespace pmacc
{

/** wrapper for native C pointer
 *
 * @tparam T_Type type of the pointed object
 */
template <typename T_Type, typename T_InitMethod = expressions::SetToNull>
class Pointer
{
public:

    typedef T_Type type;
    typedef type* PtrType;
    typedef const type* ConstPtrType;

    /** default constructor
     *
     * the default pointer points to invalid memory
     */
    HDINLINE Pointer( )
    {
        T_InitMethod()( ptr );
    }

    HDINLINE Pointer( PtrType const ptrIn ) : ptr( ptrIn )
    {
    }

    HDINLINE Pointer( const Pointer<type>& other ) : ptr( other.ptr )
    {
    }

    template<typename T_OtherInitMethod>
    HDINLINE Pointer( const Pointer<type, T_OtherInitMethod>& other ) : ptr( other.ptr )
    {
    }

    template<typename T_OtherInitMethod>
    HDINLINE Pointer& operator=(const Pointer<type, T_OtherInitMethod>& other)
    {
        ptr = other.ptr;
        return *this;
    }

    /** dereference the pointer*/
    HDINLINE type& operator*()
    {
        return *ptr;
    }

    /** dereference the pointer*/
    HDINLINE const type& operator*() const
    {
        return *ptr;
    }

    /** access member*/
    HDINLINE PtrType operator->()
    {
        return ptr;
    }

    /** access member*/
    HDINLINE ConstPtrType operator->() const
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
     * @return false if memory adress is nullptr else true
     */
    HDINLINE bool isValid( ) const
    {
        return ptr != nullptr;
    }

    PMACC_ALIGN( ptr, PtrType );
};

namespace traits
{

template<typename T_Type, typename T_InitMethod>
struct GetEmptyDefaultConstructibleType<Pointer<T_Type, T_InitMethod> >
{
    typedef Pointer<T_Type, expressions::DoNothing> type;
};
} //namespace traits

} //namespace pmacc
