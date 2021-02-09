/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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


#include "pmacc/particles/memory/boxes/TileDataBox.hpp"

#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"

namespace pmacc
{
    /**
     * Implements a Box to which elements can only be added, using atomic operations.
     *
     * @tparam TYPE datatype for addresses (must be a signed type)
     * @tparam VALUE datatype for values addresses point to
     */
    template<class TYPE, class VALUE>
    class PushDataBox : public DataBox<PitchedBox<VALUE, DIM1>>
    {
    public:
        /**
         * Constructor.
         *
         * @param data pointer to buffer holding data of type VALUE
         * @param offset relative offset to pointer start address
         * @param currentSize size of the buffer data points to
         */
        HDINLINE PushDataBox(VALUE* data, TYPE* currentSize, DataSpace<DIM1> offset = DataSpace<DIM1>(0))
            : DataBox<PitchedBox<VALUE, DIM1>>(PitchedBox<VALUE, DIM1>(data, offset))
            , currentSize(currentSize)
            , maxSize(0) /*\todo implement max size*/
        {
        }

        /** Increases the size of the stack with count elements in an atomic operation
         *
         * @warning access is only atomic within the given alpaka hierarchy
         *
         * @tparam T_Acc type of the alpaka accelerator
         * @tparam T_Hierarchy alpaka::hierarchy type of the hierarchy
         *
         * @param acc alpaka accelerator
         * @param count number of elements to increase stack with
         * @param hierarchy alpaka parallelism hierarchy levels guarantee valid
         *                  concurrency access to the memory
         *
         * @return a TileDataBox of size count pointing to the new stack elements
         */
        template<typename T_Acc, typename T_Hierarchy>
        HDINLINE TileDataBox<VALUE> pushN(T_Acc const& acc, TYPE count, T_Hierarchy const& hierarchy)
        {
            TYPE old_addr = cupla::atomicAdd(acc, currentSize, count, hierarchy);
            return TileDataBox<VALUE>(this->fixedPointer, DataSpace<DIM1>(old_addr));
        }

        /** Adds a value to the stack in an atomic operation.
         *
         * @warning access is only atomic within the given alpaka hierarchy
         *
         * @tparam T_Acc type of the alpaka accelerator
         * @tparam T_Hierarchy alpaka::hierarchy type of the hierarchy
         *
         * @param acc alpaka accelerator
         * @param val data of type VALUE to add to the stack
         * @param hierarchy alpaka parallelism hierarchy levels guarantee valid
         *                  concurrency access to the memory
         *
         * @return a TileDataBox of size count pointing to the new stack elements
         */
        template<typename T_Acc, typename T_Hierarchy>
        HDINLINE void push(T_Acc const& acc, VALUE val, T_Hierarchy const& hierarchy)
        {
            TYPE old_addr = cupla::atomicAdd(acc, currentSize, 1, hierarchy);
            (*this)[old_addr] = val;
        }

    protected:
        PMACC_ALIGN(maxSize, TYPE);
        PMACC_ALIGN(currentSize, TYPE*);
    };
} // namespace pmacc
