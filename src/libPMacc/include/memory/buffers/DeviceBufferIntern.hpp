/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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


#ifndef _DEVICEBUFFERINTERN_HPP
#define	_DEVICEBUFFERINTERN_HPP

#include <cassert>

#include "dimensions/DataSpace.hpp"
#include "memory/buffers/DeviceBuffer.hpp"
#include "memory/boxes/DataBox.hpp"

#include "eventSystem/tasks/Factory.hpp"

namespace PMacc
{

    /**
     * Internal device buffer implementation.
     */
    template <class TYPE, unsigned DIM>
    class DeviceBufferIntern : public DeviceBuffer<TYPE, DIM>
    {
    public:

        typedef typename DeviceBuffer<TYPE, DIM>::DataBoxType DataBoxType;

        /*! create device buffer
         * @param dataSpace size in any dimension of the grid on the device
         * @param sizeOnDevice memory with the current size of the grid is stored on device
         * @param useVectorAsBase use a vector as base of the array (is not lined pitched)
         *                      if true size on device is atomaticly set to false
         */
        DeviceBufferIntern(DataSpace<DIM> dataSpace, bool sizeOnDevice = false, bool useVectorAsBase = false) :
        DeviceBuffer<TYPE, DIM>(dataSpace),
        sizeOnDevice(sizeOnDevice),
        useOtherMemory(false),
        offset(DataSpace<DIM>())
        {
            //create size on device before any use of setCurrentSize
            if (useVectorAsBase)
            {
                sizeOnDevice = false;
                createSizeOnDevice(sizeOnDevice);
                createFakeData();
                this->data1D = true;
            }
            else
            {
                createSizeOnDevice(sizeOnDevice);
                createData();
                this->data1D = false;
            }

        }

        DeviceBufferIntern(DeviceBufferIntern& source, DataSpace<DIM> dataSpace, DataSpace<DIM> offset, bool sizeOnDevice = false) :
        DeviceBuffer<TYPE, DIM>(dataSpace),
        sizeOnDevice(sizeOnDevice),
        offset(offset + source.getOffset()),
        data(source.data),
        useOtherMemory(true)
        {
            createSizeOnDevice(sizeOnDevice);
            this->data1D = false;
        }

        virtual ~DeviceBufferIntern()
        {
            if (sizeOnDevice)
            {
                CUDA_CHECK(cudaFree(sizeOnDevicePtr));
            }
            if (!useOtherMemory)
            {
                CUDA_CHECK(cudaFree(data.ptr));

            }
        }

        void reset(bool preserveData = true)
        {
            this->setCurrentSize(Buffer<TYPE, DIM>::getDataSpace().getElementCount());

            __startOperation(ITask::TASK_CUDA);
            if (!preserveData)
            {
                if (DIM == DIM1)
                {
                    CUDA_CHECK(cudaMemset(data.ptr, 0, Buffer<TYPE, DIM>::getDataSpace()[0] * sizeof (TYPE)));
                }
                if (DIM == DIM2)
                {
                    CUDA_CHECK(cudaMemset2D(data.ptr, data.pitch, 0, data.xsize * sizeof (TYPE), data.ysize));
                }
                if (DIM == DIM3)
                {
                    cudaExtent extent;
                    extent.width = this->data_space[0] * sizeof (TYPE);
                    extent.height = this->data_space[1];
                    extent.depth = this->data_space[2];
                    CUDA_CHECK(cudaMemset3D(data, 0, extent));
                }
            }
        }

        DataBoxType getDataBox()
        {
            __startOperation(ITask::TASK_CUDA);
            return DataBoxType(PitchedBox<TYPE, DIM > ((TYPE*) data.ptr, offset,
                                                       this->data_space, data.pitch));
        }

        TYPE* getPointer()
        {
            __startOperation(ITask::TASK_CUDA);
            size_t widthMultHeight = 0;
            switch (DIM)
            {
            case DIM1:
                return (TYPE*) (data.ptr) + this->offset[0];
            case DIM2:
                return (TYPE*) ((char*) data.ptr + this->offset[1] * this->data.pitch) + this->offset[0];
            case DIM3:
                widthMultHeight = this->offset[1] * this->data.pitch;
                return (TYPE*) ((char*) data.ptr + this->offset[2] * widthMultHeight + widthMultHeight + this->offset[0]);
            default:
                throw std::runtime_error("DIM has invalid value");
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

        size_t* getCurrentSizeOnDevicePointer() throw (std::runtime_error)
        {
            __startOperation(ITask::TASK_CUDA);
            if (!sizeOnDevice)
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
            __startOperation(ITask::TASK_CUDA);
            return (TYPE*) data.ptr;
        }

        /*! Get current size of any dimension
         * @return count of current elements per dimension
         */
         virtual size_t getCurrentSize()
        {
            if (sizeOnDevice)
            {
                __startTransaction(__getTransactionEvent());
                Factory::getInstance().createTaskGetCurrentSizeFromDevice(*this);
                __endTransaction().waitForFinished();
            }

            return DeviceBuffer<TYPE, DIM>::getCurrentSize();
        }

         virtual void setCurrentSize(const size_t size)
        {
            Buffer<TYPE, DIM>::setCurrentSize(size);

            if (sizeOnDevice)
            {
                Factory::getInstance().createTaskSetCurrentSizeOnDevice(
                                                                        *this, size);
            }
        }
        
        void copyFrom(HostBuffer<TYPE, DIM>& other)
        {
            __startAtomicTransaction(__getTransactionEvent());
            assert(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Factory::getInstance().createTaskCopyHostToDevice(other, *this);
            __setTransactionEvent(__endTransaction());
        }

        void copyFrom(DeviceBuffer<TYPE, DIM>& other)
        {
            __startAtomicTransaction(__getTransactionEvent());
            assert(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Factory::getInstance().createTaskCopyDeviceToDevice(other, *this);
            __setTransactionEvent(__endTransaction());
        }

        const cudaPitchedPtr getCudaPitched() const
        {
            __startOperation(ITask::TASK_CUDA);
            return data;
        }

        size_t getPitch() const
        {
            return data.pitch;
        }

        virtual void setValue(TYPE value)
        {
            Factory::getInstance().createTaskSetValue(*this, value);
        };

    private:

        /*! create native array with pitched lines
         */
        void createData()
        {
            __startOperation(ITask::TASK_CUDA);
            data.ptr = NULL;
            data.pitch = 1;
            data.xsize = this->data_space[0];
            data.ysize = 1;

            if (DIM == DIM1)
            {
                log<ggLog::MEMORY >("Create device 1D data: %1% MiB") % ( this->data_space[0] * sizeof (TYPE) / 1024 / 1024 );
                CUDA_CHECK(cudaMallocPitch(&data.ptr, &data.pitch, this->data_space[0] * sizeof (TYPE), 1));
            }
            if (DIM == DIM2)
            {
                data.xsize = this->data_space[0];
                data.ysize = this->data_space[1];
                log<ggLog::MEMORY >("Create device 2D data: %1% MiB") % ( data.xsize * data.ysize * sizeof (TYPE) / 1024 / 1024 );
                CUDA_CHECK(cudaMallocPitch(&data.ptr, &data.pitch, data.xsize * sizeof (TYPE), data.ysize));

            }
            if (DIM == DIM3)
            {
                cudaExtent extent;
                extent.width = this->data_space[0] * sizeof (TYPE);
                extent.height = this->data_space[1];
                extent.depth = this->data_space[2];

                log<ggLog::MEMORY >("Create device 3D data: %1% MiB") % ( this->data_space.getElementCount() * sizeof (TYPE) / 1024 / 1024 );
                CUDA_CHECK(cudaMalloc3D(&data, extent));
            }

            reset(false);
        }

        /*!create 1D, 2D, 3D Array which use only a vector as base
         */
        void createFakeData()
        {
            __startOperation(ITask::TASK_CUDA);
            data.ptr = NULL;
            data.pitch = 1;
            data.xsize = this->data_space[0];
            data.ysize = 1;

            log<ggLog::MEMORY >("Create device fake data: %1% MiB") % ( this->data_space.getElementCount() * sizeof (TYPE) / 1024 / 1024 );
            CUDA_CHECK(cudaMallocPitch(&data.ptr, &data.pitch, this->data_space.getElementCount() * sizeof (TYPE), 1));

            //fake the pitch, thus we can use this 1D Buffer as 2D or 3D
            data.pitch = this->data_space[0] * sizeof (TYPE);

            if (DIM > DIM1)
            {
                data.ysize = this->data_space[1];
            }

            reset(false);
        }

        void createSizeOnDevice(bool sizeOnDevice)
        {
            __startOperation(ITask::TASK_HOST);
            sizeOnDevicePtr = NULL;

            if (sizeOnDevice)
            {
                CUDA_CHECK(cudaMalloc(&sizeOnDevicePtr, sizeof (size_t)));
            }
            setCurrentSize(Buffer<TYPE, DIM>::getDataSpace().getElementCount());
        }

    private:
        DataSpace<DIM> offset;

        bool sizeOnDevice;
        size_t* sizeOnDevicePtr;
        cudaPitchedPtr data;
        bool useOtherMemory;
    };

} //namespace PMacc

#endif	/* _DEVICEBUFFERINTERN_HPP */
