/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/types.hpp"

#include <limits>

namespace pmacc
{
    /**
     * Minimal function description of a buffer,
     *
     * @tparam TYPE data type stored in the buffer
     * @tparam DIM dimension of the buffer (1-3)
     */
    template<class TYPE, unsigned DIM>
    class Buffer
    {
    public:
        typedef DataBox<PitchedBox<TYPE, DIM>> DataBoxType;

        /** constructor
         *
         * @param size extent for each dimension (in elements)
         *             if the buffer is a view to an existing buffer the size
         *             can be less than `physicalMemorySize`
         * @param physicalMemorySize size of the physical memory (in elements)
         */
        Buffer(DataSpace<DIM> size, DataSpace<DIM> physicalMemorySize)
            : data_space(size)
            , data1D(true)
            , current_size(nullptr)
            , m_physicalMemorySize(physicalMemorySize)
        {
            CUDA_CHECK(cuplaMallocHost((void**) &current_size, sizeof(size_t)));
            *current_size = size.productOfComponents();
        }

        /**
         * destructor
         */
        virtual ~Buffer()
        {
            CUDA_CHECK_NO_EXCEPT(cuplaFreeHost(current_size));
        }

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
        virtual DataSpace<DIM> getDataSpace() const
        {
            return data_space;
        }

        /** get size of the physical memory (in elements)
         */
        DataSpace<DIM> getPhysicalMemorySize() const
        {
            return m_physicalMemorySize;
        }


        virtual DataSpace<DIM> getCurrentDataSpace()
        {
            return getCurrentDataSpace(getCurrentSize());
        }

        /*! Spread of memory per dimension which is currently used
         * @return if DIM == DIM1 than return count of elements (x-direction)
         * if DIM == DIM2 than return how many lines (y-direction) of memory is used
         * if DIM == DIM3 than return how many slides (z-direction) of memory is used
         */
        virtual DataSpace<DIM> getCurrentDataSpace(size_t currentSize)
        {
            DataSpace<DIM> tmp;
            int64_t current_size = static_cast<int64_t>(currentSize);

            //!\todo: current size can be changed if it is a DeviceBuffer and current size is on device
            // call first get current size (but const not allow this)

            if(DIM == DIM1)
            {
                tmp[0] = current_size;
            }
            if(DIM == DIM2)
            {
                if(current_size <= data_space[0])
                {
                    tmp[0] = current_size;
                    tmp[1] = 1;
                }
                else
                {
                    tmp[0] = data_space[0];
                    tmp[1] = (current_size + data_space[0] - 1) / data_space[0];
                }
            }
            if(DIM == DIM3)
            {
                if(current_size <= data_space[0])
                {
                    tmp[0] = current_size;
                    tmp[1] = 1;
                    tmp[2] = 1;
                }
                else if(current_size <= (data_space[0] * data_space[1]))
                {
                    tmp[0] = data_space[0];
                    tmp[1] = (current_size + data_space[0] - 1) / data_space[0];
                    tmp[2] = 1;
                }
                else
                {
                    tmp[0] = data_space[0];
                    tmp[1] = data_space[1];
                    tmp[2] = (current_size + (data_space[0] * data_space[1]) - 1) / (data_space[0] * data_space[1]);
                }
            }

            return tmp;
        }

        /*! returns the current size (count of elements)
         * @return current size
         */
        virtual size_t getCurrentSize()
        {
            __startOperation(ITask::TASK_HOST);
            return *current_size;
        }

        /*! sets the current size (count of elements)
         * @param newsize new current size
         */
        virtual void setCurrentSize(const size_t newsize)
        {
            __startOperation(ITask::TASK_HOST);
            PMACC_ASSERT(static_cast<size_t>(newsize) <= static_cast<size_t>(data_space.productOfComponents()));
            *current_size = newsize;
        }

        virtual void reset(bool preserveData = false) = 0;

        virtual void setValue(const TYPE& value) = 0;

        virtual DataBox<PitchedBox<TYPE, DIM>> getDataBox() = 0;

        inline bool is1D()
        {
            return data1D;
        }

    protected:
        /*! Check if my DataSpace is greater than other.
         * @param other other DataSpace
         * @return true if my DataSpace (one dimension) is greater than other, false otherwise
         */
        virtual bool isMyDataSpaceGreaterThan(DataSpace<DIM> other)
        {
            return !other.isOneDimensionGreaterThan(data_space);
        }

        DataSpace<DIM> data_space;
        DataSpace<DIM> m_physicalMemorySize;

        size_t* current_size;

        bool data1D;
    };

} // namespace pmacc
