/**
 * Copyright 2013 Felix Schmitt, René Widera, Wolfgang Hoenig
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
 
#ifndef _TASKCOPYDEVICETODEVICE_HPP
#define	_TASKCOPYDEVICETODEVICE_HPP

#include <cuda_runtime_api.h>

#include "types.h"

#include "eventSystem/EventSystem.hpp"
#include "eventSystem/streams/EventStream.hpp"
#include "eventSystem/tasks/StreamTask.hpp"

namespace PMacc
{

    template <class TYPE, unsigned DIM>
    class DeviceBufferIntern;

    template <class TYPE, unsigned DIM>
    class TaskCopyDeviceToDeviceBase : public StreamTask
    {
    public:

        TaskCopyDeviceToDeviceBase(const DeviceBuffer<TYPE, DIM>& src, DeviceBuffer<TYPE, DIM>& dst) :
        StreamTask()
        {
            this->source = (DeviceBufferIntern<TYPE, DIM>*) & src;
            this->destination = (DeviceBufferIntern<TYPE, DIM>*) & dst;
        }

        virtual ~TaskCopyDeviceToDeviceBase()
        {
            notify(this->myId, COPYDEVICE2DEVICE, NULL);
        }

        bool executeIntern() throw (std::runtime_error)
        {
            return isFinished();
        }

        void event(id_t eventId, EventType type, IEventData* data)
        {

        }

        virtual void init()
        {
           // __startAtomicTransaction( __getTransactionEvent());
            size_t current_size = source->getCurrentSize();
            destination->setCurrentSize(current_size);
            DataSpace<DIM> devCurrentSize = source->getCurrentDataSpace(current_size);
            if (source->is1D() && destination->is1D())
                fastCopy(source->getPointer(), destination->getPointer(), devCurrentSize.getElementCount());
            else
                copy(devCurrentSize);

            this->activate();
          //  __setTransactionEvent(__endTransaction());
        }

        std::string toString()
        {
            return "TaskCopyDeviceToDevice";
        }

    protected:

        virtual void copy(DataSpace<DIM> &devCurrentSize) = 0;

        void fastCopy(TYPE* src, TYPE* dst, size_t size)
        {
            CUDA_CHECK(cudaMemcpyAsync(dst,
                                       src,
                                       size * sizeof (TYPE), cudaMemcpyDeviceToDevice,
                                       this->getCudaStream()));
        }

        DeviceBufferIntern<TYPE, DIM> *source;
        DeviceBufferIntern<TYPE, DIM> *destination;
    };


    template <class TYPE, unsigned DIM>
    class TaskCopyDeviceToDevice;

    template <class TYPE>
    class TaskCopyDeviceToDevice<TYPE, DIM1> : public TaskCopyDeviceToDeviceBase<TYPE, DIM1>
    {
    public:

        TaskCopyDeviceToDevice(const DeviceBuffer<TYPE, DIM1>& src, DeviceBuffer<TYPE, DIM1>& dst) :
        TaskCopyDeviceToDeviceBase<TYPE, DIM1>(src, dst)
        {
        }

    private:

        virtual void copy(DataSpace<DIM1> &devCurrentSize)
        {

            CUDA_CHECK(cudaMemcpyAsync(this->destination->getPointer(),
                                       this->source->getPointer(),
                                       devCurrentSize[0] * sizeof (TYPE), cudaMemcpyDeviceToDevice,
                                       this->getCudaStream()));
        }

    };

    template <class TYPE>
    class TaskCopyDeviceToDevice<TYPE, DIM2> : public TaskCopyDeviceToDeviceBase<TYPE, DIM2>
    {
    public:

        TaskCopyDeviceToDevice(const DeviceBuffer<TYPE, DIM2>& src, DeviceBuffer<TYPE, DIM2>& dst) :
        TaskCopyDeviceToDeviceBase<TYPE, DIM2>(src, dst)
        {
        }

    private:

        virtual void copy(DataSpace<DIM2> &devCurrentSize)
        {
            CUDA_CHECK(cudaMemcpy2DAsync(this->destination->getPointer(),
                                         this->destination->getPitch(),
                                         this->source->getPointer(),
                                         this->source->getPitch(),
                                         devCurrentSize[0] * sizeof (TYPE),
                                         devCurrentSize[1],
                                         cudaMemcpyDeviceToDevice,
                                         this->getCudaStream()));

        }

    };

    template <class TYPE>
    class TaskCopyDeviceToDevice<TYPE, DIM3> : public TaskCopyDeviceToDeviceBase<TYPE, DIM3>
    {
    public:

        TaskCopyDeviceToDevice(const DeviceBuffer<TYPE, DIM3>& src, DeviceBuffer<TYPE, DIM3>& dst) :
        TaskCopyDeviceToDeviceBase<TYPE, DIM3>(src, dst)
        {
        }

    private:

        virtual void copy(DataSpace<DIM3> &devCurrentSize)
        {

            cudaMemcpy3DParms params;

            //  assert(this->source->getDataSpace().getElementCount() <= this->destination->getDataSpace().getElementCount());

            params.srcArray = NULL;
            params.srcPos = make_cudaPos(
                                         this->source->getOffset()[0] * sizeof (TYPE),
                                         this->source->getOffset()[1],
                                         this->source->getOffset()[2]);
            params.srcPtr = this->source->getCudaPitched();

            params.dstArray = NULL;
            params.dstPos = make_cudaPos(
                                         this->destination->getOffset()[0] * sizeof (TYPE),
                                         this->destination->getOffset()[1],
                                         this->destination->getOffset()[2]);
            ;
            params.dstPtr = this->destination->getCudaPitched();

            params.extent = make_cudaExtent(
                                            devCurrentSize[0] * sizeof (TYPE),
                                            devCurrentSize[1],
                                            devCurrentSize[2]);
            params.kind = cudaMemcpyDeviceToDevice;
            CUDA_CHECK(cudaMemcpy3DAsync(&params, this->getCudaStream()));
        }

    };

} //namespace PMacc


#endif	/* _TASKCOPYDEVICETODEVICE_HPP */

