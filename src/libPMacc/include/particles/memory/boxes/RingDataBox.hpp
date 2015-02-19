/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, Rene Widera
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

#ifndef RINGDATABOX_HPP
#define	RINGDATABOX_HPP

#include <cuda.h>

#include "particles/memory/boxes/TileDataBox.hpp"

#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

namespace PMacc
{


/**
 * A Box which allows adding (at the end) and removing elements (from the front) in atomic operations.
 *
 * @tparam TYPE is the type of addresses
 * @tparam VALUE is the type of data addresses point to
 */
template<class TYPE, class VALUE>
class RingDataBox : public DataBox<PitchedBox<VALUE, DIM1> >
{
private:
    typedef DataBox<PitchedBox<TYPE, DIM1> > IndexBoxType;

    enum
    {
        PUSH, POP, ERR
    };
public:

    /** default constructor
     *
     * ATTENTION: after this call the object is in a invalid state and must
     * initialized with a assignment of a valid RingDataBox
     */
    HINLINE RingDataBox()
    {
    }

    /**
     * Constructor.
     *
     * @param data pointer to a buffer of type VALUE
     * @param size size of the buffer data points to. size must be at least two-times the number of threads
     *  concurrently accessing data
     * @param pointerBegin pointer to the first elements of the buffer
     * @param indexBox[PUSH] pointer to the last element of the buffer
     */
    HDINLINE RingDataBox(VALUE *data, TYPE size, IndexBoxType indexBox) :
    DataBox<PitchedBox<VALUE, DIM1> >(PitchedBox<VALUE, DIM1>(data, DataSpace<DIM1>(0))),
    indexBox(indexBox),
    size(size)
    {
    }

    /**
     * Adds an element at the end of the buffer in an atomic operation.
     *
     * @param val element of type VALUE to add
     */
    HDINLINE void push(VALUE val)
    {
#if !defined(__CUDA_ARCH__) // Host code path
        const TYPE old_idx = (indexBox[PUSH]);
        old_idx >= size - 1 ? (indexBox[PUSH]) = 0 : (indexBox[PUSH]) = old_idx + 1;
#else
        const TYPE old_idx = atomicInc(&(indexBox[PUSH]), size - 1);
#endif
        (*this)[old_idx] = val;
    }

    /**
     * Removes an element from the front of the buffer in an atomic operation.
     *
     * @return the element of type VALUE removed from the buffer
     */
    HDINLINE VALUE &pop()
    {
#if (__CUDA_ARCH__>=200)
        const TYPE push_idx = indexBox[PUSH]; // == B
#endif

        //!\todo check if we can use atomicInc
#if !defined(__CUDA_ARCH__) // Host code path
        const TYPE old_idx = (indexBox[POP]);
        old_idx >= size - 1 ? (indexBox[POP]) = 0 : (indexBox[POP]) = old_idx + 1;

#else
        const TYPE old_idx = atomicInc(&(indexBox[POP]), size - 1);
#endif
#if (__CUDA_ARCH__>=200)
        /*old_idx == F*/
        const TYPE new_idx = (old_idx + 1) % size; //==F'

        const bool a = (old_idx > push_idx);
        const bool b = (old_idx < push_idx);

        const bool c = (new_idx >= push_idx);
        const bool d = (new_idx <= push_idx);

        const bool e = (new_idx < old_idx);
        const bool f = !(e); //F'>=F

        const bool overflow = (b && c && f) || (e && ((a && c) || (b && d)));


        if (overflow)
            printf("Ringbuffer: memory overflow\n");
#endif

        return (*this)[old_idx];
    }

    /*HDINLINE TileDataBox<VALUE> pushN(TYPE count);

    HDINLINE  TileDataBox<VALUE> popN(TYPE count);*/

protected:
    PMACC_ALIGN(indexBox, IndexBoxType);
    PMACC_ALIGN(size, TYPE);
};
}


#endif	/* RINGDATABOX_HPP */
