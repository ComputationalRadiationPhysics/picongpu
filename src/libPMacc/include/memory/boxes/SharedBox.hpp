/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include <cuSTL/cursor/compile-time/BufferCursor.hpp>
#include <math/vector/Float.hpp>
#include <math/Vector.hpp>
#include "types.h"

namespace PMacc
{

/** create shared memory on gpu
 *
 * @tparam T_TYPE type of memory objects
 * @tparam T_Vector CT::Vector with size description (per dimension)
 * @tparam T_id unique id for this object
 *              (is needed if more than one instance of shared memory in one kernel is used)
 * @tparam T_dim dimension of the memory (supports DIM1,DIM2 and DIM3)
 */
template<typename T_TYPE, class T_Vector, uint32_t T_id=0, uint32_t T_dim=T_Vector::dim>
class SharedBox;

template<typename T_TYPE, class T_Vector, uint32_t T_id>
class SharedBox<T_TYPE, T_Vector, T_id, DIM1>
{
public:

    enum
    {
        Dim = DIM1
    };
    typedef T_TYPE ValueType;
    typedef ValueType& RefValueType;
    typedef T_Vector Size;
    typedef SharedBox<ValueType, math::CT::Int<Size::x::value>, T_id> ReducedType;
    typedef SharedBox<ValueType, T_Vector, T_id, DIM1> This;

    HDINLINE RefValueType operator[](const int idx)
    {
        return fixedPointer[idx];
    }

    HDINLINE RefValueType operator[](const int idx) const
    {
        return fixedPointer[idx];
    }

    HDINLINE SharedBox(ValueType* pointer) :
    fixedPointer(pointer)
    {
    }

    DINLINE SharedBox() :
    fixedPointer(NULL)
    {
    }

    /*!return the first value in the box (list)
     * @return first value
     */
    HDINLINE RefValueType operator*()
    {
        return *(fixedPointer);
    }

    HDINLINE ValueType const * getPointer() const
    {
        return fixedPointer;
    }
    HDINLINE ValueType* getPointer()
    {
        return fixedPointer;
    }

    /* This call synchronizes a block and must be called from all threads and not inside a if clauses*/
    static DINLINE This init()
    {
        __shared__ ValueType mem_sh[Size::x::value];
        return This((ValueType*) mem_sh);
    }

protected:

    PMACC_ALIGN(fixedPointer, ValueType*);
};

template<typename T_TYPE, class T_Vector, uint32_t T_id>
class SharedBox<T_TYPE, T_Vector,T_id, DIM2 >
{
public:

    enum
    {
        Dim = DIM2
    };
    typedef T_TYPE ValueType;
    typedef ValueType& RefValueType;
    typedef T_Vector Size;
    typedef SharedBox<ValueType, math::CT::Int<Size::x::value>, T_id > ReducedType;
    typedef SharedBox<ValueType, T_Vector, T_id, DIM2 > This;

    HDINLINE SharedBox(ValueType* pointer = NULL) :
    fixedPointer(pointer)
    {
    }

    HDINLINE ReducedType operator[](const int idx)
    {
        return ReducedType(this->fixedPointer + idx * Size::x::value);
    }

    HDINLINE ReducedType operator[](const int idx) const
    {
        return ReducedType(this->fixedPointer + idx * Size::x::value);
    }

    /*!return the first value in the box (list)
     * @return first value
     */
    HDINLINE RefValueType operator*()
    {
        return *((ValueType*) fixedPointer);
    }

    HDINLINE ValueType const * getPointer() const
    {
        return fixedPointer;
    }
    HDINLINE ValueType* getPointer()
    {
        return fixedPointer;
    }

    /* This call synchronizes a block and must be called from all threads and not inside a if clauses*/
    static DINLINE This init()
    {
        __shared__ ValueType mem_sh[Size::y::value][Size::x::value];
        return This((ValueType*) mem_sh);
    }

    HDINLINE PMacc::cursor::CT::BufferCursor<ValueType, ::PMacc::math::CT::Int<sizeof (ValueType) * Size::x::value> >
    toCursor() const
    {
        return PMacc::cursor::CT::BufferCursor<ValueType, ::PMacc::math::CT::Int<sizeof (ValueType) * Size::x::value> >
            ((ValueType*) fixedPointer);
    }

protected:

    PMACC_ALIGN(fixedPointer, ValueType*);
};

template<typename T_TYPE, class T_Vector, uint32_t T_id>
class SharedBox<T_TYPE, T_Vector, T_id, DIM3>
{
public:

    enum
    {
        Dim = DIM3
    };
    typedef T_TYPE ValueType;
    typedef ValueType& RefValueType;
    typedef T_Vector Size;
    typedef SharedBox<ValueType, math::CT::Int<Size::x::value, Size::y::value>, T_id > ReducedType;
    typedef SharedBox<ValueType, T_Vector, T_id, DIM3 > This;

    HDINLINE ReducedType operator[](const int idx)
    {
        return ReducedType(this->fixedPointer + idx *  (Size::x::value * Size::y::value));
    }

    HDINLINE ReducedType operator[](const int idx) const
    {
        return ReducedType(this->fixedPointer + idx *  (Size::x::value *Size::y::value));
    }

    HDINLINE SharedBox(ValueType* pointer = NULL) :
    fixedPointer(pointer)
    {
    }

    /*!return the first value in the box (list)
     * @return first value
     */
    HDINLINE RefValueType operator*()
    {
        return *(fixedPointer);
    }

    HDINLINE ValueType const * getPointer() const
    {
        return fixedPointer;
    }
    HDINLINE ValueType* getPointer()
    {
        return fixedPointer;
    }

    HDINLINE PMacc::cursor::CT::BufferCursor<ValueType, ::PMacc::math::CT::Int<sizeof (ValueType) * Size::x::value,
    sizeof (ValueType) * Size::x::value * Size::y::value> >
    toCursor() const
    {
        return PMacc::cursor::CT::BufferCursor<ValueType, ::PMacc::math::CT::Int<sizeof (ValueType) * Size::x::value,
            sizeof (ValueType) * Size::x::value * Size::y::value> >
            ((ValueType*)fixedPointer);
    }

    /*this call synchronize a block and must called from any thread and not inside a if clauses*/
    static DINLINE This init()
    {
        __shared__ ValueType mem_sh[Size::z::value][Size::y::value][Size::x::value];
        return This((ValueType*) mem_sh);
    }

protected:

    PMACC_ALIGN(fixedPointer, ValueType*);

};


}

