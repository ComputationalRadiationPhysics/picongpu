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
 
#ifndef _TASKCOPYHOSTTODEVICE_HPP
#define	_TASKCOPYHOSTTODEVICE_HPP

#include <cuda_runtime_api.h>

#include "eventSystem/EventSystem.hpp"
#include "eventSystem/streams/EventStream.hpp"
#include "eventSystem/tasks/StreamTask.hpp"

namespace PMacc
{

    template <class TYPE, unsigned DIM>
    class HostBufferIntern;
    template <class TYPE, unsigned DIM>
    class DeviceBufferIntern;

    template <class TYPE, unsigned DIM>
    class TaskCopyHostToDeviceBase : public StreamTask
    {
    public:

        TaskCopyHostToDeviceBase(const HostBuffer<TYPE, DIM>& src, DeviceBuffer<TYPE, DIM>& dst) :
        StreamTask()
        {
            this->host = (HostBufferIntern<TYPE, DIM>*) & src;
            this->device = (DeviceBufferIntern<TYPE, DIM>*) & dst;
        }

        virtual ~TaskCopyHostToDeviceBase()
        {
            notify(this->myId, COPYHOST2DEVICE, NULL);
            //std::cout<<"destructor TaskH2D"<<std::endl;
        }

        bool executeIntern()
        {
            return isFinished();
        }

        void event(id_t eventId, EventType type, IEventData* data)
        {
        }

        virtual void init()
        {
         //   __startAtomicTransaction(__getTransactionEvent());
            size_t current_size = host->getCurrentSize();
            DataSpace<DIM> hostCurrentSize = host->getCurrentDataSpace(current_size);
            if (host->is1D() && device->is1D())
                fastCopy(host->getPointer(), device->getPointer(), hostCurrentSize.getElementCount());
            else
                copy(hostCurrentSize);
            device->setCurrentSize(current_size);
            this->activate();
         //   __setTransactionEvent(__endTransaction());
        }

        std::string toString()
        {
            return "TaskCopyHostToDevice";
        }


    protected:

        virtual void copy(DataSpace<DIM> &hostCurrentSize) = 0;

        void fastCopy(TYPE* src, TYPE* dst, size_t size)
        {
            CUDA_CHECK(cudaMemcpyAsync(dst,
                                       src,
                                       size * sizeof (TYPE),
                                       cudaMemcpyHostToDevice,
                                       this->getCudaStream()));
            // std::cout<<"-----------fast H2D"<<std::endl;;
        }


        HostBufferIntern<TYPE, DIM> *host;
        DeviceBufferIntern<TYPE, DIM> *device;

    };

    template <class TYPE, unsigned DIM>
    class TaskCopyHostToDevice;

    template <class TYPE>
    class TaskCopyHostToDevice<TYPE, DIM1> : public TaskCopyHostToDeviceBase<TYPE, DIM1>
    {
    public:

        TaskCopyHostToDevice(const HostBuffer<TYPE, DIM1>& src, DeviceBuffer<TYPE, DIM1>& dst) :
        TaskCopyHostToDeviceBase<TYPE, DIM1>(src, dst)
        {
        }
    private:

        virtual void copy(DataSpace<DIM1> &hostCurrentSize)
        {
            CUDA_CHECK(cudaMemcpyAsync(this->device->getPointer(), /*pointer include X offset*/
                                       this->host->getBasePointer(),
                                       hostCurrentSize[0] * sizeof (TYPE), cudaMemcpyHostToDevice,
                                       this->getCudaStream()));
        }
    };

    template <class TYPE>
    class TaskCopyHostToDevice<TYPE, DIM2> : public TaskCopyHostToDeviceBase<TYPE, DIM2>
    {
    public:

        TaskCopyHostToDevice(const HostBuffer<TYPE, DIM2>& src, DeviceBuffer<TYPE, DIM2>& dst) :
        TaskCopyHostToDeviceBase<TYPE, DIM2>(src, dst)
        {
        }
    private:

        virtual void copy(DataSpace<DIM2> &hostCurrentSize)
        {
            CUDA_CHECK(cudaMemcpy2DAsync(this->device->getPointer(),
                                         this->device->getPitch(), /*this is pitch*/
                                         this->host->getBasePointer(),
                                         this->host->getDataSpace()[0] * sizeof (TYPE), /*this is pitch*/
                                         hostCurrentSize[0] * sizeof (TYPE),
                                         hostCurrentSize[1],
                                         cudaMemcpyHostToDevice,
                                         this->getCudaStream()));
        }
    };

    template <class TYPE>
    class TaskCopyHostToDevice<TYPE, DIM3> : public TaskCopyHostToDeviceBase<TYPE, DIM3>
    {
    public:

        TaskCopyHostToDevice(const HostBuffer<TYPE, DIM3>& src, DeviceBuffer<TYPE, DIM3>& dst) :
        TaskCopyHostToDeviceBase<TYPE, DIM3>(src, dst)
        {
        }
    private:

        virtual void copy(DataSpace<DIM3> &hostCurrentSize)
        {
            cudaPitchedPtr hostPtr;
            hostPtr.pitch = this->host->getDataSpace()[0] * sizeof (TYPE);
            hostPtr.ptr = this->host->getBasePointer();
            hostPtr.xsize = this->host->getDataSpace()[0] * sizeof (TYPE);
            hostPtr.ysize = this->host->getDataSpace()[1];

            cudaMemcpy3DParms params = {0};
            params.dstArray = NULL;
            params.dstPos = make_cudaPos(this->device->getOffset()[0] * sizeof (TYPE),
                                         this->device->getOffset()[1],
                                         this->device->getOffset()[2]);
            params.dstPtr = this->device->getCudaPitched();

            params.srcArray = NULL;
            params.srcPos = make_cudaPos(0, 0, 0);
            params.srcPtr = hostPtr;

            params.extent = make_cudaExtent(
                                            hostCurrentSize[0] * sizeof (TYPE),
                                            hostCurrentSize[1],
                                            hostCurrentSize[2]);
            params.kind = cudaMemcpyHostToDevice;

            CUDA_CHECK(cudaMemcpy3DAsync(&params, this->getCudaStream()));
        }
    };


} //namespace PMacc


#endif	/* _TASKCOPYHOSTTODEVICE_HPP */

