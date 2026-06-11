/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once


#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/particles/memory/boxes/TileDataBox.hpp"

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
        HDINLINE PushDataBox(VALUE* data, TYPE* currentSize, DataSpace<DIM1> offset = {})
            : DataBox<PitchedBox<VALUE, DIM1>>(
                  DataBox<PitchedBox<VALUE, DIM1>>{PitchedBox<VALUE, DIM1>(data)}.shift(offset))
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
        template<typename T_Worker, typename T_Hierarchy>
        HDINLINE TileDataBox<VALUE> pushN(T_Worker const& worker, TYPE count, T_Hierarchy const& hierarchy)
        {
            TYPE old_addr = alpaka::atomicAdd(worker.getAcc(), currentSize, count, hierarchy);
            return TileDataBox<VALUE>(this->m_ptr, DataSpace<DIM1>(old_addr));
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
        template<typename T_Worker, typename T_Hierarchy>
        HDINLINE void push(T_Worker const& worker, VALUE val, T_Hierarchy const& hierarchy)
        {
            TYPE old_addr = alpaka::atomicAdd(worker.getAcc(), currentSize, 1, hierarchy);
            (*this)[old_addr] = val;
        }

    protected:
        PMACC_ALIGN(currentSize, TYPE*);
        PMACC_ALIGN(maxSize, TYPE);
    };
} // namespace pmacc
