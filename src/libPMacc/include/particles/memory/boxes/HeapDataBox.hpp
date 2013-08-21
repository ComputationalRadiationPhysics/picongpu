/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera
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
 
#ifndef HEAPDATABOX_HPP
#define	HEAPDATABOX_HPP

#include "particles/memory/boxes/RingDataBox.hpp"

#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

namespace PMacc
{


/**
 * Represents an abstraction between actual data in memory and their indizes in a RingDataBox.
 *
 * @tparam TYPE datatype for indizes
 * @tparam VALUE datatype for indexed data
 */
template<class TYPE, class VALUE>
class HeapDataBox : public DataBox<PitchedBox<VALUE, DIM1> >
{
public:

    typedef VALUE Type;

    /**
     * Constructor.
     *
     * @param data pointer to buffer of type VALUE
     * @param virtualMemory a RingDataBox with sizes corresponding to data, representing indizes of data's VALUEs
     */
    HDINLINE HeapDataBox(VALUE *data, RingDataBox<TYPE, TYPE> virtualMemory) :
    DataBox<PitchedBox<VALUE, DIM1> >(PitchedBox<VALUE, DIM1>(data, DataSpace<DIM1>())),
    virtualMemory(virtualMemory)
    {

    }

    /**
     * Pops the last index from virtual memory and returns a reference to the VALUE at this index.
     *
     * @return reference to VALUE at top-most index
     */
    HDINLINE VALUE &pop()
    {
        TYPE addr = virtualMemory.pop();
        return (*this)[addr];
    }

    HDINLINE TYPE popIdx()
    {
        return virtualMemory.pop();
    }

    /**
     * Computes the index of the VALUE at address old and pushes it to virtual memory.
     * 
     * @param old address of VALUE of which the index should be pushed
     */
    HDINLINE void push(VALUE &old)
    {

        //const double x = (double) sizeof (VALUE);
       // TYPE index = floor((double) ((size_t) (&old) - (size_t)this->fixedPointer) / x + 0.00001);
        TYPE index = ((size_t) (&old) - (size_t) ((VALUE*)(this->fixedPointer))) / sizeof(VALUE);
        virtualMemory.push(index);

        /*
         TYPE index = ((size_t) (&old) - (size_t) ((VALUE*)(this->fixedPointer))) / sizeof(VALUE);
         virtualMemory.push(index);*/
    }

    HDINLINE void pushIdx(const TYPE idx)
    {
        virtualMemory.push(idx);
    }

protected:
    PMACC_ALIGN8(virtualMemory,RingDataBox<TYPE, TYPE>);
};

}

#endif	/* HEAPDATABOX_HPP */
