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
 

#ifndef _BUFFER_HPP
#define	_BUFFER_HPP

#include <cassert>
#include <limits>

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

namespace PMacc
{

    /**
     * Minimal function descriptiin of a buffer,
     * 
     * @tparam TYPE datatype stored in the buffer
     * @tparam DIM dimension of the buffer (1-3)
     */
    template <class TYPE, unsigned DIM>
    class Buffer
    {
    public:

        typedef DataBox<PitchedBox<TYPE, DIM> > DataBoxType;
        
        /**
         * constructor
         * @param dataSpace description of spread of any dimension
         */
        Buffer(DataSpace<DIM> dataSpace);

        /**
         * destructor
         */
        virtual ~Buffer();

        /*! Get base pointer to memory
         * @return pointer to this buffer in memory
         */
        virtual TYPE* getBasePointer() = 0;

        /*! Get pointer that includes all offsets
         * @return pointer to a point in a memory array
         */
        virtual TYPE* getPointer() = 0;

        /*! Get max spread (elements) of any dimension
         * @return spread (elements) per dimension
         */
        virtual DataSpace<DIM> getDataSpace() const;

        virtual DataSpace<DIM> getCurrentDataSpace();

        /*! Spread of memory per dimension which is currently used
         * @return if DIM == DIM1 than return count of elements (x-direction)
         * if DIM == DIM2 than return how many lines (y-direction) of memory is used
         * if DIM == DIM3 than return how many slides (z-direction) of memory is used
         */
        virtual DataSpace<DIM> getCurrentDataSpace(size_t currentSize) ;

        /*! returns the current size (count of elements)
         * @return current size
         */
        virtual size_t getCurrentSize();

        /*! sets the current size (count of elements)
         * @param newsize new current size
         */
        virtual void setCurrentSize(size_t newsize);

        virtual void reset(bool preserveData = false) = 0;

        virtual void setValue(const TYPE& value) = 0;

        virtual DataBox<PitchedBox<TYPE,DIM> > getDataBox() = 0;
        
        inline bool is1D();

    protected:

        /*! Check if my DataSpace is greater than other.
         * @param other other DataSpace
         * @return true if my DataSpace (one dimension) is greater than other, false otherwise
         */
        virtual bool isMyDataSpaceGreaterThan(DataSpace<DIM> other);

        DataSpace<DIM> data_space;

        size_t *current_size;
        
        bool data1D;

    };

} //namespace PMacc

#endif	/* _BUFFER_HPP */

