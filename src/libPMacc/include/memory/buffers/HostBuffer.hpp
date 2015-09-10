/**
 * Copyright 2013-2015 Rene Widera, Benjamin Worpitz, Alexander Grund
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

#include <cuSTL/container/HostBuffer.hpp>
#include "memory/buffers/Buffer.hpp"
#include "dimensions/DataSpace.hpp"

namespace PMacc
{

    class EventTask;

    template <class TYPE, unsigned DIM>
    class DeviceBuffer;

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

        __forceinline__
        container::HostBuffer<TYPE, DIM>
        cartBuffer()
        {
            container::HostBuffer<TYPE, DIM> result;
            result.dataPointer = this->getBasePointer();
            result._size = math::Size_t<DIM>(this->getDataSpace());
            if(DIM >= 2)
                result.pitch[0] = result._size.x() * sizeof(TYPE);
            if(DIM >= 3)
                result.pitch[1] = result.pitch[0] * result._size.y();
            result.refCount = new int;
            *result.refCount = 2;
            return result;
        }

    protected:

        /**
         * Constructor.
         *
         * @param dataSpace size of each dimension of the buffer
         */
        HostBuffer(DataSpace<DIM> dataSpace) :
        Buffer<TYPE, DIM>(dataSpace)
        {

        }
    };

} //namespace PMacc
