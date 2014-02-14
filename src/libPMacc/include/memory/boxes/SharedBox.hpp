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
#include "dimensions/TVec.h"


#include <cuSTL/cursor/compile-time/BufferCursor.hpp>    
#include <math/vector/Float.hpp>


namespace PMacc
{

template<typename TYPE, class TVector>
class SharedBox;

template<typename TYPE, uint32_t _X_>
class SharedBox<TYPE, TVec<_X_> >
{
public:

    enum
    {
        Dim = DIM1
    };
    typedef TYPE ValueType;
    typedef ValueType& RefValueType;
    typedef SharedBox<TYPE, TVec<_X_> > ReducedType;

    HDINLINE RefValueType operator[](const int idx)
    {
        return fixedPointer[idx];
    }

    HDINLINE RefValueType operator[](const int idx) const
    {
        return fixedPointer[idx];
    }

    HDINLINE SharedBox(TYPE* pointer) :
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

    HDINLINE TYPE* getPointer()
    {
        return fixedPointer;
    }

    /*this call synchronize a block and must called from any thread and not inside a if clauses*/
    static DINLINE SharedBox<TYPE, TVec<_X_> > init()
    {
        __shared__ TYPE mem_sh[_X_];
        __syncthreads(); /*wait that all shared memory is initialised*/
        return SharedBox<TYPE, TVec<_X_> >((TYPE*) mem_sh);
    }

protected:

    PMACC_ALIGN(fixedPointer, TYPE*);
};

template<typename TYPE, uint32_t _X_, uint32_t _Y_>
class SharedBox<TYPE, TVec<_X_, _Y_> >
{
public:

    enum
    {
        Dim = DIM2
    };
    typedef TYPE ValueType;
    typedef ValueType& RefValueType;
    typedef SharedBox<TYPE, TVec<_X_> > ReducedType;

    HDINLINE SharedBox(TYPE* pointer = NULL) :
    fixedPointer(pointer)
    {
    }

    HDINLINE ReducedType operator[](const int idx)
    {
        return ReducedType(this->fixedPointer + idx * (int) (_X_));
    }

    HDINLINE ReducedType operator[](const int idx) const
    {
        return ReducedType(this->fixedPointer + idx * (int) (_X_));
    }

    /*!return the first value in the box (list)
     * @return first value
     */
    HDINLINE RefValueType operator*()
    {
        return *((TYPE*) fixedPointer);
    }

    HDINLINE TYPE* getPointer()
    {
        return fixedPointer;
    }

    /*this call synchronize a block and must called from any thread and not inside a if clauses*/
    static DINLINE SharedBox<TYPE, TVec<_X_, _Y_> > init()
    {
        __shared__ TYPE mem_sh[_Y_][_X_];
        __syncthreads(); /*wait that all shared memory is initialised*/
        return SharedBox<TYPE, TVec<_X_, _Y_> >((TYPE*) mem_sh);
    }

    HDINLINE PMacc::cursor::CT::BufferCursor<TYPE, ::PMacc::math::CT::Int<sizeof (TYPE) * _X_> >
    toCursor() const
    {
        return PMacc::cursor::CT::BufferCursor<TYPE, ::PMacc::math::CT::Int<sizeof (TYPE) * _X_> >
            ((TYPE*) fixedPointer);
    }

protected:

    PMACC_ALIGN(fixedPointer, TYPE*);
};

template<typename TYPE, uint32_t _X_, uint32_t _Y_, uint32_t _Z_>
class SharedBox<TYPE, TVec<_X_, _Y_, _Z_> >
{
public:

    enum
    {
        Dim = DIM3
    };
    typedef TYPE ValueType;
    typedef ValueType& RefValueType;
    typedef SharedBox<TYPE, TVec<_X_, _Y_> > ReducedType;

    HDINLINE ReducedType operator[](const int idx)
    {
        return ReducedType(this->fixedPointer + idx * (int) (_X_ * _Y_));
    }

    HDINLINE ReducedType operator[](const int idx) const
    {
        return ReducedType(this->fixedPointer + idx * (int) (_X_ * _Y_));
    }

    HDINLINE SharedBox(TYPE* pointer = NULL) :
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

    HDINLINE TYPE* getPointer()
    {
        return fixedPointer;
    }

    HDINLINE PMacc::cursor::CT::BufferCursor<TYPE, ::PMacc::math::CT::Int<sizeof (TYPE) * _X_,
    sizeof (TYPE) * _X_ * _Y_> >
    toCursor() const
    {
        return PMacc::cursor::CT::BufferCursor<TYPE, ::PMacc::math::CT::Int<sizeof (TYPE) * _X_,
            sizeof (TYPE) * _X_ * _Y_> >
            ((TYPE*)fixedPointer);
    }

    /*this call synchronize a block and must called from any thread and not inside a if clauses*/
    static DINLINE SharedBox<TYPE, TVec<_X_, _Y_, _Z_> > init()
    {
        __shared__ TYPE mem_sh[_Z_][_Y_][_X_];
        __syncthreads(); /*wait that all shared memory is initialised*/
        return SharedBox<TYPE, TVec<_X_, _Y_, _Z_> >((TYPE*) mem_sh);
    }

protected:

    PMACC_ALIGN(fixedPointer, TYPE*);

};


}

