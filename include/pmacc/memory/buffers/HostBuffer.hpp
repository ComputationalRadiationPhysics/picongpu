/* Copyright 2013-2018 Rene Widera, Benjamin Worpitz, Alexander Grund
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


#include "pmacc/cuSTL/container/HostBuffer.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"
#include "pmacc/dimensions/DataSpace.hpp"

namespace pmacc
{

    class EventTask;

    template <class TYPE, unsigned DIM>
    class DeviceBuffer;

    template <class TYPE, unsigned DIM>
    class Buffer;

    /**
     * Interface for a DIM-dimensional Buffer of type TYPE on the host
     *
     * @tparam TYPE datatype for buffer data
     * @tparam DIM dimension of the buffer
     */
    template <class TYPE, unsigned DIM>
    class HostBuffer : public Buffer<TYPE, DIM>
    {
    public:
        /**
         * Copies the data from the given DeviceBuffer to this HostBuffer.
         *
         * @param other DeviceBuffer to copy data from
         */
        virtual void copyFrom(DeviceBuffer<TYPE, DIM>& other) = 0;

        /**
         * Returns the current size pointer.
         *
         * @return pointer to current size
         */
        virtual size_t* getCurrentSizePointer()
        {
            __startOperation(ITask::TASK_HOST);
            return this->current_size;
        }

        /**
         * Destructor.
         */
        virtual ~HostBuffer()
        {
        };

        HINLINE
        container::HostBuffer<TYPE, DIM>
        cartBuffer()
        {
            math::Size_t<DIM - 1> pitch;
            if(DIM >= 2)
                pitch[0] = this->getPhysicalMemorySize()[0] * sizeof(TYPE);
            if(DIM == 3)
                pitch[1] = pitch[0] * this->getPhysicalMemorySize()[1];
            container::HostBuffer<TYPE, DIM> result(this->getBasePointer(), this->getDataSpace(), false, pitch);
            return result;
        }

    protected:

        /** Constructor.
         *
         * @param size extent for each dimension (in elements)
         *             if the buffer is a view to an existing buffer the size
         *             can be less than `physicalMemorySize`
         * @param physicalMemorySize size of the physical memory (in elements)
         */
        HostBuffer(DataSpace<DIM> size, DataSpace<DIM> physicalMemorySize) :
        Buffer<TYPE, DIM>(size, physicalMemorySize)
        {

        }
    };

} //namespace pmacc
