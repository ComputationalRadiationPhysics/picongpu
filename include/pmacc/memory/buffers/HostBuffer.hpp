/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz, Alexander Grund
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
    template<class TYPE, unsigned DIM>
    class HostBuffer;

    namespace detail
    {
        template<class TYPE>
        container::HostBuffer<TYPE, 1u> make_CartBuffer(HostBuffer<TYPE, 1u>& hb)
        {
            return container::HostBuffer<TYPE, 1u>(hb.getBasePointer(), hb.getDataSpace(), false);
        }

        template<class TYPE>
        container::HostBuffer<TYPE, 2u> make_CartBuffer(HostBuffer<TYPE, 2u>& hb)
        {
            math::Size_t<2u - 1u> pitch;
            pitch[0] = hb.getPhysicalMemorySize()[0] * sizeof(TYPE);
            return container::HostBuffer<TYPE, 2u>(hb.getBasePointer(), hb.getDataSpace(), false, pitch);
        }

        template<class TYPE>
        container::HostBuffer<TYPE, 3u> make_CartBuffer(HostBuffer<TYPE, 3u>& hb)
        {
            math::Size_t<3u - 1u> pitch;
            pitch[0] = hb.getPhysicalMemorySize()[0] * sizeof(TYPE);
            pitch[1] = pitch[0] * hb.getPhysicalMemorySize()[1];
            return container::HostBuffer<TYPE, 3u>(hb.getBasePointer(), hb.getDataSpace(), false, pitch);
        }
    } // namespace detail
    class EventTask;

    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class Buffer;

    /**
     * Interface for a DIM-dimensional Buffer of type TYPE on the host
     *
     * @tparam TYPE datatype for buffer data
     * @tparam DIM dimension of the buffer
     */
    template<class TYPE, unsigned DIM>
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
        virtual ~HostBuffer(){};

        /**
         * Conversion to cuSTL HostBuffer.
         *
         * Returns a cuSTL HostBuffer with reference to the same data.
         */
        HINLINE
        container::HostBuffer<TYPE, DIM> cartBuffer()
        {
            return detail::make_CartBuffer(*this);
        }

    protected:
        /** Constructor.
         *
         * @param size extent for each dimension (in elements)
         *             if the buffer is a view to an existing buffer the size
         *             can be less than `physicalMemorySize`
         * @param physicalMemorySize size of the physical memory (in elements)
         */
        HostBuffer(DataSpace<DIM> size, DataSpace<DIM> physicalMemorySize)
            : Buffer<TYPE, DIM>(size, physicalMemorySize)
        {
        }
    };

} // namespace pmacc
