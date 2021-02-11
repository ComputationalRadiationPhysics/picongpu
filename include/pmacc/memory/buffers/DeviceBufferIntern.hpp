/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/assert.hpp"

#include <memory>


namespace pmacc
{
    /**
     * Internal device buffer implementation.
     */
    template<class TYPE, unsigned DIM>
    class DeviceBufferIntern : public DeviceBuffer<TYPE, DIM>
    {
    public:
        typedef typename DeviceBuffer<TYPE, DIM>::DataBoxType DataBoxType;

        /*! create device buffer
         * @param size extent for each dimension (in elements)
         * @param sizeOnDevice memory with the current size of the grid is stored on device
         * @param useVectorAsBase use a vector as base of the array (is not lined pitched)
         *                      if true size on device is atomaticly set to false
         */
        DeviceBufferIntern(DataSpace<DIM> size, bool sizeOnDevice = false, bool useVectorAsBase = false)
            : DeviceBuffer<TYPE, DIM>(size, size)
            , sizeOnDevice(sizeOnDevice)
            , useOtherMemory(false)
            , offset(DataSpace<DIM>())
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

        DeviceBufferIntern(
            DeviceBuffer<TYPE, DIM>& source,
            DataSpace<DIM> size,
            DataSpace<DIM> offset,
            bool sizeOnDevice = false)
            : DeviceBuffer<TYPE, DIM>(size, source.getPhysicalMemorySize())
            , sizeOnDevice(sizeOnDevice)
            , offset(offset + source.getOffset())
            , data(source.getCudaPitched())
            , useOtherMemory(true)
        {
            createSizeOnDevice(sizeOnDevice);
            this->data1D = false;
        }

        virtual ~DeviceBufferIntern()
        {
            __startOperation(ITask::TASK_DEVICE);

            if(sizeOnDevice)
            {
                CUDA_CHECK_NO_EXCEPT(cuplaFree(sizeOnDevicePtr));
            }
            if(!useOtherMemory)
            {
                CUDA_CHECK_NO_EXCEPT(cuplaFree(data.ptr));
            }
        }

        void reset(bool preserveData = true)
        {
            this->setCurrentSize(Buffer<TYPE, DIM>::getDataSpace().productOfComponents());

            __startOperation(ITask::TASK_DEVICE);
            if(!preserveData)
            {
                // Using Array is a workaround for types without default constructor
                memory::Array<TYPE, 1> tmp;
                memset(reinterpret_cast<void*>(tmp.data()), 0, sizeof(tmp));
                // use first element to avoid issue because Array is aligned (sizeof can be larger than component type)
                setValue(tmp[0]);
            }
        }

        DataBoxType getDataBox()
        {
            __startOperation(ITask::TASK_DEVICE);
            return DataBoxType(
                PitchedBox<TYPE, DIM>((TYPE*) data.ptr, offset, this->getPhysicalMemorySize(), data.pitch));
        }

        TYPE* getPointer()
        {
            __startOperation(ITask::TASK_DEVICE);

            if(DIM == DIM1)
            {
                return (TYPE*) (data.ptr) + this->offset[0];
            }
            else if(DIM == DIM2)
            {
                return (TYPE*) ((char*) data.ptr + this->offset[1] * this->data.pitch) + this->offset[0];
            }
            else
            {
                const size_t offsetY = this->offset[1] * this->data.pitch;
                const size_t sizePlaneXY = this->getPhysicalMemorySize()[1] * this->data.pitch;
                return (TYPE*) ((char*) data.ptr + this->offset[2] * sizePlaneXY + offsetY) + this->offset[0];
            }
        }

        DataSpace<DIM> getOffset() const
        {
            return offset;
        }

        bool hasCurrentSizeOnDevice() const
        {
            return sizeOnDevice;
        }

        size_t* getCurrentSizeOnDevicePointer()
        {
            __startOperation(ITask::TASK_DEVICE);
            if(!sizeOnDevice)
            {
                throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
            }
            return sizeOnDevicePtr;
        }

        size_t* getCurrentSizeHostSidePointer()
        {
            __startOperation(ITask::TASK_HOST);
            return this->current_size;
        }

        TYPE* getBasePointer()
        {
            __startOperation(ITask::TASK_DEVICE);
            return (TYPE*) data.ptr;
        }

        /*! Get current size of any dimension
         * @return count of current elements per dimension
         */
        virtual size_t getCurrentSize()
        {
            if(sizeOnDevice)
            {
                __startTransaction(__getTransactionEvent());
                Environment<>::get().Factory().createTaskGetCurrentSizeFromDevice(*this);
                __endTransaction().waitForFinished();
            }

            return DeviceBuffer<TYPE, DIM>::getCurrentSize();
        }

        virtual void setCurrentSize(const size_t size)
        {
            Buffer<TYPE, DIM>::setCurrentSize(size);

            if(sizeOnDevice)
            {
                Environment<>::get().Factory().createTaskSetCurrentSizeOnDevice(*this, size);
            }
        }

        void copyFrom(HostBuffer<TYPE, DIM>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopyHostToDevice(other, *this);
        }

        void copyFrom(DeviceBuffer<TYPE, DIM>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopyDeviceToDevice(other, *this);
        }

        const cuplaPitchedPtr getCudaPitched() const
        {
            __startOperation(ITask::TASK_DEVICE);
            return data;
        }

        size_t getPitch() const
        {
            return data.pitch;
        }

        virtual void setValue(const TYPE& value)
        {
            Environment<>::get().Factory().createTaskSetValue(*this, value);
        };

    private:
        /*! create native array with pitched lines
         */
        void createData()
        {
            __startOperation(ITask::TASK_DEVICE);
            data.ptr = nullptr;
            data.pitch = 1;
            data.xsize = this->getDataSpace()[0] * sizeof(TYPE);
            data.ysize = 1;

            if(DIM == DIM1)
            {
                log<ggLog::MEMORY>("Create device 1D data: %1% MiB") % (data.xsize / 1024 / 1024);
                CUDA_CHECK(cuplaMallocPitch(&data.ptr, &data.pitch, data.xsize, 1));
            }
            if(DIM == DIM2)
            {
                data.ysize = this->getDataSpace()[1];
                log<ggLog::MEMORY>("Create device 2D data: %1% MiB") % (data.xsize * data.ysize / 1024 / 1024);
                CUDA_CHECK(cuplaMallocPitch(&data.ptr, &data.pitch, data.xsize, data.ysize));
            }
            if(DIM == DIM3)
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
            __startOperation(ITask::TASK_DEVICE);
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

            if(DIM > DIM1)
            {
                data.ysize = this->getDataSpace()[1];
            }

            reset(false);
        }

        void createSizeOnDevice(bool sizeOnDevice)
        {
            __startOperation(ITask::TASK_HOST);
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

    /** Factory for a new heap-allocated DeviceBufferIntern buffer object that is a deep copy of the given device
     * buffer
     *
     * @tparam TYPE value type
     * @tparam DIM index dimensionality
     *
     * @param source source device buffer
     */
    template<class TYPE, unsigned DIM>
    HINLINE std::unique_ptr<DeviceBufferIntern<TYPE, DIM>> makeDeepCopy(DeviceBuffer<TYPE, DIM>& source)
    {
        auto result = std::make_unique<DeviceBufferIntern<TYPE, DIM>>(
            source,
            source.getDataSpace(),
            DataSpace<DIM>::create(0));
        result->copyFrom(source);
        // Wait for copy to finish, so that the resulting object is safe to use after return
        __getTransactionEvent().waitForFinished();
        return result;
    }

} // namespace pmacc
