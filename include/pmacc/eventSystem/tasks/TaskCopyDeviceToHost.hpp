/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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

#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/eventSystem/streams/EventStream.hpp"
#include "pmacc/eventSystem/tasks/StreamTask.hpp"


#include <iomanip>

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class HostBuffer;
    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class TaskCopyDeviceToHostBase : public StreamTask
    {
    public:
        TaskCopyDeviceToHostBase(DeviceBuffer<TYPE, DIM>& src, HostBuffer<TYPE, DIM>& dst) : StreamTask()
        {
            this->host = &dst;
            this->device = &src;
        }

        virtual ~TaskCopyDeviceToHostBase()
        {
            notify(this->myId, COPYDEVICE2HOST, nullptr);
        }

        bool executeIntern()
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*)
        {
        }

        std::string toString()
        {
            return "TaskCopyDeviceToHost";
        }

        virtual void init()
        {
            size_t current_size = device->getCurrentSize();
            host->setCurrentSize(current_size);
            DataSpace<DIM> devCurrentSize = device->getCurrentDataSpace(current_size);
            if(host->is1D() && device->is1D())
                fastCopy(device->getPointer(), host->getPointer(), devCurrentSize.productOfComponents());
            else
                copy(devCurrentSize);

            this->activate();
        }

    protected:
        virtual void copy(DataSpace<DIM>& devCurrentSize) = 0;

        void fastCopy(TYPE* src, TYPE* dst, size_t size)
        {
            CUDA_CHECK(
                cuplaMemcpyAsync(dst, src, size * sizeof(TYPE), cuplaMemcpyDeviceToHost, this->getCudaStream()));
        }

        HostBuffer<TYPE, DIM>* host;
        DeviceBuffer<TYPE, DIM>* device;
    };

    template<class TYPE, unsigned DIM>
    class TaskCopyDeviceToHost;

    template<class TYPE>
    class TaskCopyDeviceToHost<TYPE, DIM1> : public TaskCopyDeviceToHostBase<TYPE, DIM1>
    {
    public:
        TaskCopyDeviceToHost(DeviceBuffer<TYPE, DIM1>& src, HostBuffer<TYPE, DIM1>& dst)
            : TaskCopyDeviceToHostBase<TYPE, DIM1>(src, dst)
        {
        }

    private:
        virtual void copy(DataSpace<DIM1>& devCurrentSize)
        {
            CUDA_CHECK(cuplaMemcpyAsync(
                this->host->getBasePointer(),
                this->device->getPointer(),
                devCurrentSize[0] * sizeof(TYPE),
                cuplaMemcpyDeviceToHost,
                this->getCudaStream()));
        }
    };

    template<class TYPE>
    class TaskCopyDeviceToHost<TYPE, DIM2> : public TaskCopyDeviceToHostBase<TYPE, DIM2>
    {
    public:
        TaskCopyDeviceToHost(DeviceBuffer<TYPE, DIM2>& src, HostBuffer<TYPE, DIM2>& dst)
            : TaskCopyDeviceToHostBase<TYPE, DIM2>(src, dst)
        {
        }

    private:
        virtual void copy(DataSpace<DIM2>& devCurrentSize)
        {
            CUDA_CHECK(cuplaMemcpy2DAsync(
                this->host->getBasePointer(),
                this->host->getDataSpace()[0] * sizeof(TYPE), /*this is pitch*/
                this->device->getPointer(),
                this->device->getPitch(), /*this is pitch*/
                devCurrentSize[0] * sizeof(TYPE),
                devCurrentSize[1],
                cuplaMemcpyDeviceToHost,
                this->getCudaStream()));
        }
    };

    template<class TYPE>
    class TaskCopyDeviceToHost<TYPE, DIM3> : public TaskCopyDeviceToHostBase<TYPE, DIM3>
    {
    public:
        TaskCopyDeviceToHost(DeviceBuffer<TYPE, DIM3>& src, HostBuffer<TYPE, DIM3>& dst)
            : TaskCopyDeviceToHostBase<TYPE, DIM3>(src, dst)
        {
        }

    private:
        virtual void copy(DataSpace<DIM3>& devCurrentSize)
        {
            cuplaPitchedPtr hostPtr;
            hostPtr.pitch = this->host->getDataSpace()[0] * sizeof(TYPE);
            hostPtr.ptr = this->host->getBasePointer();
            hostPtr.xsize = this->host->getDataSpace()[0] * sizeof(TYPE);
            hostPtr.ysize = this->host->getDataSpace()[1];

            cuplaMemcpy3DParms params;
            params.srcArray = nullptr;
            params.srcPos = make_cuplaPos(
                this->device->getOffset()[0] * sizeof(TYPE),
                this->device->getOffset()[1],
                this->device->getOffset()[2]);
            params.srcPtr = this->device->getCudaPitched();

            params.dstArray = nullptr;
            params.dstPos = make_cuplaPos(0, 0, 0);
            params.dstPtr = hostPtr;

            params.extent = make_cuplaExtent(devCurrentSize[0] * sizeof(TYPE), devCurrentSize[1], devCurrentSize[2]);
            params.kind = cuplaMemcpyDeviceToHost;

            CUDA_CHECK(cuplaMemcpy3DAsync(&params, this->getCudaStream()));
        }
    };

} // namespace pmacc
