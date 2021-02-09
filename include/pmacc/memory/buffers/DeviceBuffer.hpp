/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
 *                     Alexander Grund
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


#include "pmacc/cuSTL/container/view/View.hpp"
#include "pmacc/cuSTL/container/DeviceBuffer.hpp"
#include "pmacc/math/vector/Int.hpp"
#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"
#include "pmacc/types.hpp"


#include <stdexcept>

namespace pmacc
{
    class EventTask;

    template<class TYPE, unsigned DIM>
    class HostBuffer;

    template<class TYPE, unsigned DIM>
    class Buffer;

    /**
     * Interface for a DIM-dimensional Buffer of type TYPE on the device.
     *
     * @tparam TYPE datatype of the buffer
     * @tparam DIM dimension of the buffer
     */
    template<class TYPE, unsigned DIM>
    class DeviceBuffer : public Buffer<TYPE, DIM>
    {
    protected:
        /** constructor
         *
         * @param size extent for each dimension (in elements)
         *             if the buffer is a view to an existing buffer the size
         *             can be less than `physicalMemorySize`
         * @param physicalMemorySize size of the physical memory (in elements)
         */
        DeviceBuffer(DataSpace<DIM> size, DataSpace<DIM> physicalMemorySize)
            : Buffer<TYPE, DIM>(size, physicalMemorySize)
        {
        }

    public:
        using Buffer<TYPE, DIM>::setCurrentSize; //!\todo :this function was hidden, I don't know why.

        /**
         * Destructor.
         */
        virtual ~DeviceBuffer(){};

        HINLINE
        container::CartBuffer<
            TYPE,
            DIM,
            allocator::DeviceMemAllocator<TYPE, DIM>,
            copier::D2DCopier<DIM>,
            assigner::DeviceMemAssigner<>>
        cartBuffer() const
        {
            cuplaPitchedPtr cuplaData = this->getCudaPitched();
            math::Size_t<DIM - 1> pitch;
            if(DIM >= 2)
                pitch[0] = cuplaData.pitch;
            if(DIM == 3)
                pitch[1] = pitch[0] * this->getPhysicalMemorySize()[1];
            container::DeviceBuffer<TYPE, DIM> result((TYPE*) cuplaData.ptr, this->getDataSpace(), false, pitch);
            return result;
        }

        /**
         * Returns offset of elements in every dimension.
         *
         * @return count of elements
         */
        virtual DataSpace<DIM> getOffset() const = 0;

        /**
         * Show if current size is stored on device.
         *
         * @return return false if no size is stored on device, true otherwise
         */
        virtual bool hasCurrentSizeOnDevice() const = 0;

        /**
         * Returns pointer to current size on device.
         *
         * @return pointer which point to device memory of current size
         */
        virtual size_t* getCurrentSizeOnDevicePointer() = 0;

        /** Returns host pointer of current size storage
         *
         * @return pointer to stored value on host side
         */
        virtual size_t* getCurrentSizeHostSidePointer() = 0;

        /**
         * Sets current size of any dimension.
         *
         * If stream is 0, this function is blocking (we use a kernel to set size).
         * Keep in mind: on Fermi-architecture, kernels in different streams may run at the same time
         * (only used if size is on device).
         *
         * @param size count of elements per dimension
         */
        virtual void setCurrentSize(const size_t size) = 0;

        /**
         * Returns the internal pitched cupla pointer.
         *
         * @return internal pitched cupla pointer
         */
        virtual const cuplaPitchedPtr getCudaPitched() const = 0;

        /** get line pitch of memory in byte
         *
         * @return size of one line in memory
         */
        virtual size_t getPitch() const = 0;

        /**
         * Copies data from the given HostBuffer to this DeviceBuffer.
         *
         * @param other the HostBuffer to copy from
         */
        virtual void copyFrom(HostBuffer<TYPE, DIM>& other) = 0;

        /**
         * Copies data from the given DeviceBuffer to this DeviceBuffer.
         *
         * @param other the DeviceBuffer to copy from
         */
        virtual void copyFrom(DeviceBuffer<TYPE, DIM>& other) = 0;
    };

} // namespace pmacc
