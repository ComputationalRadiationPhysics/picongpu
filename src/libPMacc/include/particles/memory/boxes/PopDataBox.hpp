/**
 * Copyright 2013-2015 Felix Schmitt, Heiko Burau, Rene Widera,
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
 * Implements a stack Box from which elements can only be removed, using atomic operations.
 *
 * @tparam TYPE datatype for addresses (must be a signed type)
 * @tparam VALUE datatype for values addresses point to
 */
template<class TYPE, class VALUE>
class PopDataBox : public DataBox<PitchedBox<VALUE, DIM1> >
{
public:

    /**
     * Constructor.
     *
     * @param data pointer to buffer holding data of type VALUE
     * @param currentSize pointer to size of buffer
     * @param maxSize maximum number of elements this box holds
     * @param offset relative offset to pointer start adress
     */
    HDINLINE PopDataBox(VALUE *data, TYPE *currentSize, TYPE maxSize, DataSpace<DIM1> offset = DataSpace<DIM1>(0)) :
    DataBox<PitchedBox<VALUE, DIM1> >(PitchedBox<VALUE, DIM1> (data, offset)),
    currentSize(currentSize), maxSize(maxSize)
    {
    }

    /**
     * Removes count elements from the stack in an atomic operation.
     *
     * \todo This method unse int32_t and limits the element count to INT_MAX
     *
     * @param count number of elements to pop from stack
     * @return a TileDataBox of type VALUE with count elements
     */
    HDINLINE TileDataBox<VALUE> popN(TYPE count)
    {
#if !defined(__CUDA_ARCH__) // Host code path
        int32_t old_addr = (*currentSize);
        (*currentSize) -= count;
#else
        int32_t old_addr = (int32_t) atomicSub((int32_t*) currentSize, (int32_t) count);
#endif

        if (old_addr <= 0)
        {
            *currentSize = 0;
            return TileDataBox<VALUE > (this->fixedPointer, DataSpace<DIM1 > (0), 0);
        }

        if (old_addr < (int32_t) count)
        {
            *currentSize = 0;
            return TileDataBox<VALUE > (this->fixedPointer, DataSpace<DIM1 > (0), old_addr);
        }

        return TileDataBox<VALUE > (this->fixedPointer, DataSpace<DIM1 > (old_addr - count), count);
    }

    /**
     * Removes an element from the stack in an atomic operation.
     *
     * @return the element which has been removed
     */

    /* \todo not working if we have no elements on stack*/
    HDINLINE VALUE &pop()
    {
#if !defined(__CUDA_ARCH__) // Host code path
        TYPE old_addr = --(*currentSize);
#else
        TYPE old_addr = atomicSub(currentSize, 1) - 1;
#endif
        return (*this)[old_addr];
    }


protected:

    PMACC_ALIGN(maxSize, TYPE);
    // ptr must be in device-memory
    PMACC_ALIGN(currentSize, TYPE*);
};

}
