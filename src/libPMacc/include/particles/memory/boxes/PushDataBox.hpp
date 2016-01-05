/**
 * Copyright 2013-2016 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz
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
#include <cuda.h>

#include "particles/memory/boxes/TileDataBox.hpp"

#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

namespace PMacc
{


    /**
     * Implements a Box to which elements can only be added, using atomic operations.
     *
     * @tparam TYPE datatype for addresses (must be a signed type)
     * @tparam VALUE datatype for values addresses point to
     */
    template<class TYPE, class VALUE>
    class PushDataBox : public DataBox<PitchedBox<VALUE, DIM1> >
    {
    public:

        /**
         * Constructor.
         *
         * @param data pointer to buffer holding data of type VALUE
         * @param offset relative offset to pointer start address
         * @param currentSize size of the buffer data points to
         */
        HDINLINE PushDataBox(VALUE *data, TYPE *currentSize, DataSpace<DIM1> offset=DataSpace<DIM1>(0)) :
        DataBox<PitchedBox<VALUE, DIM1> >(PitchedBox<VALUE,DIM1> ( data, offset)),
        currentSize(currentSize),maxSize(0) /*\todo implement max size*/
        {

        }

        /**
         * Increases the size of the stack with count elements in an atomic operation.
         *
         * @param count number of elements to increase stack with
         * @return a TileDataBox of size count pointing to the new stack elements
         */
        HDINLINE TileDataBox<VALUE> pushN(TYPE count)
        {
#if !defined(__CUDA_ARCH__) // Host code path
            //TYPE old_addr = (*currentSize) = (*currentSize) + count;
            //old_addr -= count;
            TYPE old_addr = (*currentSize);
            *currentSize += count;
#else
            TYPE old_addr = atomicAdd(currentSize, count);
#endif
            return TileDataBox<VALUE > (this->fixedPointer, DataSpace<DIM1>(old_addr));
        }

        /**
         * Adds val to the stack in an atomic operation.
         *
         * @param val data of type VALUE to add to the stack
         */
        HDINLINE void push(VALUE val)
        {
#if !defined(__CUDA_ARCH__) // Host code path
            TYPE old_addr = (*currentSize)++;
#else
            TYPE old_addr = atomicAdd(currentSize, 1);
#endif
            (*this)[old_addr] = val;
        }

    protected:
        PMACC_ALIGN(maxSize,TYPE);
        PMACC_ALIGN(currentSize,TYPE*);
    };
}
