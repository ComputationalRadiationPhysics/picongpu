/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include <cuSTL/container/view/View.hpp>
#include <cuSTL/container/DeviceBuffer.hpp>
#include <math/vector/Int.hpp>
#include <math/vector/Size_t.hpp>
#include <memory/buffers/Buffer.hpp>
#include <types.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdexcept>

namespace PMacc
{
    class EventTask;

    template <class TYPE, unsigned DIM>
    class HostBuffer;

    /**
     * Interface for a DIM-dimensional Buffer of type TYPE on the device.
     *
     * @tparam TYPE datatype of the buffer
     * @tparam DIM dimension of the buffer
     */
    template <class TYPE, unsigned DIM>
    class DeviceBuffer : public Buffer<TYPE, DIM>
    {
    protected:

        DeviceBuffer(DataSpace<DIM> dataSpace) :
        Buffer<TYPE, DIM>(dataSpace)
        {

        }

    public:

        using Buffer<TYPE, DIM>::setCurrentSize; //!\todo :this function was hidden, I don't know why.

        /**
         * Destructor.
         */
        virtual ~DeviceBuffer()
        {
        };


#define COMMA ,

        __forceinline__
        container::CartBuffer<TYPE, DIM, allocator::DeviceMemAllocator<TYPE, DIM>,
                                copier::D2DCopier<DIM>,
                                assigner::DeviceMemAssigner<DIM> >
        cartBuffer() const
        {
            container::DeviceBuffer<TYPE, DIM> result;
            cudaPitchedPtr cudaData = this->getCudaPitched();
            result.dataPointer = (TYPE*)cudaData.ptr;
            result._size = (math::Size_t<DIM>)this->getDataSpace();
            if(DIM == 2) result.pitch[0] = cudaData.pitch;
            if(DIM == 3)
            {
                result.pitch[0] = cudaData.pitch;
                result.pitch[1] = cudaData.pitch * result._size.y();
            }
#ifndef __CUDA_ARCH__
            result.refCount = new int;
#endif
            *result.refCount = 2;
            return result;
        }
#undef COMMA


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
        virtual size_t* getCurrentSizeHostSidePointer()=0;

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
         * Returns the internal pitched cuda pointer.
         *
         * @return internal pitched cuda pointer
         */
        virtual const cudaPitchedPtr getCudaPitched() const = 0;

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

} //namespace PMacc
