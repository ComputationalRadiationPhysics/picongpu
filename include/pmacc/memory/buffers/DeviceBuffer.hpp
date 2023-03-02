/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
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

#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"
#include "pmacc/types.hpp"

#include <memory>

namespace pmacc
{
    class EventTask;

    template<class TYPE, unsigned DIM>
    class HostBuffer;

    template<class TYPE, unsigned DIM>
    class Buffer;

    /** DIM-dimensional Buffer of type TYPE on the device.
     *
     * @tparam TYPE datatype of the buffer
     * @tparam DIM dimension of the buffer
     */
    template<class TYPE, unsigned DIM>
    class DeviceBuffer : public Buffer<TYPE, DIM>
    {
    public:
        using DataBoxType = typename Buffer<TYPE, DIM>::DataBoxType;

        /** Create device buffer
         *
         * Allocate new memory on the device.
         *
         * @param size extent for each dimension (in elements)
         * @param sizeOnDevice memory with the current size of the grid is stored on device
         * @param useVectorAsBase use a vector as base of the array (is not lined pitched)
         *                      if true size on device is atomaticly set to false
         */
        DeviceBuffer(DataSpace<DIM> size, bool sizeOnDevice = false, bool useVectorAsBase = false)
            : Buffer<TYPE, DIM>(size, size)
            , offset(DataSpace<DIM>())
            , sizeOnDevice(sizeOnDevice)
            , useOtherMemory(false)
        {
            // create size on device before any use of setCurrentSize
            if(useVectorAsBase)
            {
                this->sizeOnDevice = false;
                createSizeOnDevice(this->sizeOnDevice);
                createFakeData();
                this->data1D = true;
            }
            else
            {
                createSizeOnDevice(this->sizeOnDevice);
                createData();
                this->data1D = false;
            }
        }

        /** Create a shallow copy of the given source buffer
         *
         * The resulting buffer is effectively a subview to the source buffer.
         *
         * @param source source device buffer
         * @param size extent for each dimension (in elements)
         * @param offset extra offset in the source buffer
         * @param sizeOnDevice memory with the current size of the grid is stored on device
         */
        DeviceBuffer(
            DeviceBuffer<TYPE, DIM>& source,
            DataSpace<DIM> size,
            DataSpace<DIM> offset,
            bool sizeOnDevice = false)
            : Buffer<TYPE, DIM>(size, source.getPhysicalMemorySize())
            , offset(offset + source.getOffset())
            , sizeOnDevice(sizeOnDevice)
            , data(source.getCudaPitched())
            , useOtherMemory(true)
        {
            createSizeOnDevice(sizeOnDevice);
            this->data1D = false;
        }

        ~DeviceBuffer() override
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);

            if(sizeOnDevice)
            {
                CUDA_CHECK_NO_EXCEPT(cuplaFree(sizeOnDevicePtr));
            }
            if(!useOtherMemory)
            {
                CUDA_CHECK_NO_EXCEPT(cuplaFree(data.ptr));
            }
        }

        void reset(bool preserveData = true) override
        {
            this->setCurrentSize(Buffer<TYPE, DIM>::getDataSpace().productOfComponents());

            eventSystem::startOperation(ITask::TASK_DEVICE);
            if(!preserveData)
            {
                // Using Array is a workaround for types without default constructor
                memory::Array<TYPE, 1> tmp;
                memset(reinterpret_cast<void*>(tmp.data()), 0, sizeof(tmp));
                // use first element to avoid issue because Array is aligned (sizeof can be larger than component type)
                setValue(tmp[0]);
            }
        }

        DataBoxType getDataBox() override
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return DataBoxType(PitchedBox<TYPE, DIM>((TYPE*) data.ptr, this->getPhysicalMemorySize(), data.pitch))
                .shift(offset);
        }

        TYPE* getPointer() override
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);

            if constexpr(DIM == DIM1)
            {
                return (TYPE*) (data.ptr) + this->offset[0];
            }
            else if constexpr(DIM == DIM2)
            {
                return (TYPE*) ((char*) data.ptr + this->offset[1] * this->data.pitch) + this->offset[0];
            }

            // path for the highest supported dimension DIM3
            const size_t offsetY = this->offset[1] * this->data.pitch;
            const size_t sizePlaneXY = this->getPhysicalMemorySize()[1] * this->data.pitch;
            return (TYPE*) ((char*) data.ptr + this->offset[2] * sizePlaneXY + offsetY) + this->offset[0];
        }

        /**
         * Returns offset of elements in every dimension.
         *
         * @return count of elements
         */
        DataSpace<DIM> getOffset() const
        {
            return offset;
        }

        /**
         * Show if current size is stored on device.
         *
         * @return return false if no size is stored on device, true otherwise
         */
        bool hasCurrentSizeOnDevice() const
        {
            return sizeOnDevice;
        }

        /**
         * Returns pointer to current size on device.
         *
         * @return pointer which point to device memory of current size
         */
        size_t* getCurrentSizeOnDevicePointer()
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            if(!sizeOnDevice)
            {
                throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
            }
            return sizeOnDevicePtr;
        }

        /** Returns host pointer of current size storage
         *
         * @return pointer to stored value on host side
         */
        size_t* getCurrentSizeHostSidePointer()
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return this->current_size;
        }

        TYPE* getBasePointer() override
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return (TYPE*) data.ptr;
        }

        /*! Get current size of any dimension
         * @return count of current elements per dimension
         */
        size_t getCurrentSize() override
        {
            if(sizeOnDevice)
            {
                eventSystem::startTransaction(eventSystem::getTransactionEvent());
                Environment<>::get().Factory().createTaskGetCurrentSizeFromDevice(*this);
                eventSystem::endTransaction().waitForFinished();
            }

            return Buffer<TYPE, DIM>::getCurrentSize();
        }

        /**
         * Sets current size of any dimension.
         *
         * If stream is 0, this function is blocking (we use a kernel to set size).
         * Keep in mind: on Fermi-architecture, kernels in different streams may run at the same time
         * (only used if size is on device).
         *
         * @param size count of elements per dimension
         */
        void setCurrentSize(const size_t size)
        {
            Buffer<TYPE, DIM>::setCurrentSize(size);

            if(sizeOnDevice)
            {
                Environment<>::get().Factory().createTaskSetCurrentSizeOnDevice(*this, size);
            }
        }

        /**
         * Copies data from the given HostBuffer to this DeviceBuffer.
         *
         * @param other the HostBuffer to copy from
         */
        void copyFrom(HostBuffer<TYPE, DIM>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopyHostToDevice(other, *this);
        }

        /**
         * Copies data from the given DeviceBuffer to this DeviceBuffer.
         *
         * @param other the DeviceBuffer to copy from
         */
        void copyFrom(DeviceBuffer<TYPE, DIM>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopyDeviceToDevice(other, *this);
        }

        /**
         * Returns the internal pitched cupla pointer.
         *
         * @return internal pitched cupla pointer
         */
        const cuplaPitchedPtr getCudaPitched() const
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            return data;
        }

        /** get line pitch of memory in byte
         *
         * @return size of one line in memory
         */
        size_t getPitch() const
        {
            return data.pitch;
        }

        void setValue(const TYPE& value) override
        {
            Environment<>::get().Factory().createTaskSetValue(*this, value);
        };

    private:
        /*! create native array with pitched lines
         */
        void createData()
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            data.ptr = nullptr;
            data.pitch = 1;
            data.xsize = this->getDataSpace()[0] * sizeof(TYPE);
            data.ysize = 1;

            if constexpr(DIM == DIM1)
            {
                log<ggLog::MEMORY>("Create device 1D data: %1% MiB") % (data.xsize / 1024 / 1024);
                CUDA_CHECK(cuplaMallocPitch(&data.ptr, &data.pitch, data.xsize, 1));
            }
            if constexpr(DIM == DIM2)
            {
                data.ysize = this->getDataSpace()[1];
                log<ggLog::MEMORY>("Create device 2D data: %1% MiB") % (data.xsize * data.ysize / 1024 / 1024);
                CUDA_CHECK(cuplaMallocPitch(&data.ptr, &data.pitch, data.xsize, data.ysize));
            }
            if constexpr(DIM == DIM3)
            {
                cuplaExtent extent;
                extent.width = this->getDataSpace()[0] * sizeof(TYPE);
                extent.height = this->getDataSpace()[1];
                extent.depth = this->getDataSpace()[2];

                log<ggLog::MEMORY>("Create device 3D data: %1% MiB")
                    % (this->getDataSpace().productOfComponents() * sizeof(TYPE) / 1024 / 1024);
                CUDA_CHECK(cuplaMalloc3D(&data, extent));
            }

            reset(false);
        }

        /*!create 1D, 2D, 3D Array which use only a vector as base
         */
        void createFakeData()
        {
            eventSystem::startOperation(ITask::TASK_DEVICE);
            data.ptr = nullptr;
            data.pitch = 1;
            data.xsize = this->getDataSpace()[0] * sizeof(TYPE);
            data.ysize = 1;

            log<ggLog::MEMORY>("Create device fake data: %1% MiB")
                % (this->getDataSpace().productOfComponents() * sizeof(TYPE) / 1024 / 1024);
            CUDA_CHECK(cuplaMallocPitch(
                &data.ptr,
                &data.pitch,
                this->getDataSpace().productOfComponents() * sizeof(TYPE),
                1));

            // fake the pitch, thus we can use this 1D Buffer as 2D or 3D
            data.pitch = this->getDataSpace()[0] * sizeof(TYPE);

            if constexpr(DIM > DIM1)
            {
                data.ysize = this->getDataSpace()[1];
            }

            reset(false);
        }

        void createSizeOnDevice(bool sizeOnDevice)
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            sizeOnDevicePtr = nullptr;

            if(sizeOnDevice)
            {
                CUDA_CHECK(cuplaMalloc((void**) &sizeOnDevicePtr, sizeof(size_t)));
            }
            setCurrentSize(this->getDataSpace().productOfComponents());
        }

    private:
        DataSpace<DIM> offset;

        bool sizeOnDevice;
        size_t* sizeOnDevicePtr;
        cuplaPitchedPtr data;
        bool useOtherMemory;
    };

    /** Factory for a new heap-allocated DeviceBuffer buffer object that is a deep copy of the given device
     * buffer
     *
     * @tparam TYPE value type
     * @tparam DIM index dimensionality
     *
     * @param source source device buffer
     */
    template<class TYPE, unsigned DIM>
    HINLINE std::unique_ptr<DeviceBuffer<TYPE, DIM>> makeDeepCopy(DeviceBuffer<TYPE, DIM>& source)
    {
        // We have to call this constructor to allocate a new data storage and not shallow-copy the source
        auto result = std::make_unique<DeviceBuffer<TYPE, DIM>>(source.getDataSpace());
        result->copyFrom(source);
        // Wait for copy to finish, so that the resulting object is safe to use after return
        eventSystem::getTransactionEvent().waitForFinished();
        return result;
    }

} // namespace pmacc
