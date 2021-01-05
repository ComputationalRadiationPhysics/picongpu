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


namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class HostBuffer;
    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class TaskCopyHostToDeviceBase : public StreamTask
    {
    public:
        TaskCopyHostToDeviceBase(HostBuffer<TYPE, DIM>& src, DeviceBuffer<TYPE, DIM>& dst) : StreamTask()
        {
            this->host = &src;
            this->device = &dst;
        }

        virtual ~TaskCopyHostToDeviceBase()
        {
            notify(this->myId, COPYHOST2DEVICE, nullptr);
        }

        bool executeIntern()
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*)
        {
        }

        virtual void init()
        {
            size_t current_size = host->getCurrentSize();
            DataSpace<DIM> hostCurrentSize = host->getCurrentDataSpace(current_size);
            /* IMPORTENT: `setCurrentSize()` must be called before the native cupla memcopy
             * is called else `setCurrentSize()` is not handled as part of this task.
             * The reason for that is that the native memcopy calls `this->getCudaStream()`
             * but not register an task before this `init()` is finished.
             */
            device->setCurrentSize(current_size);
            if(host->is1D() && device->is1D())
                fastCopy(host->getPointer(), device->getPointer(), hostCurrentSize.productOfComponents());
            else
                copy(hostCurrentSize);

            this->activate();
        }

        std::string toString()
        {
            return "TaskCopyHostToDevice";
        }


    protected:
        virtual void copy(DataSpace<DIM>& hostCurrentSize) = 0;

        void fastCopy(TYPE* src, TYPE* dst, size_t size)
        {
            CUDA_CHECK(
                cuplaMemcpyAsync(dst, src, size * sizeof(TYPE), cuplaMemcpyHostToDevice, this->getCudaStream()));
        }


        HostBuffer<TYPE, DIM>* host;
        DeviceBuffer<TYPE, DIM>* device;
    };

    template<class TYPE, unsigned DIM>
    class TaskCopyHostToDevice;

    template<class TYPE>
    class TaskCopyHostToDevice<TYPE, DIM1> : public TaskCopyHostToDeviceBase<TYPE, DIM1>
    {
    public:
        TaskCopyHostToDevice(HostBuffer<TYPE, DIM1>& src, DeviceBuffer<TYPE, DIM1>& dst)
            : TaskCopyHostToDeviceBase<TYPE, DIM1>(src, dst)
        {
        }

    private:
        virtual void copy(DataSpace<DIM1>& hostCurrentSize)
        {
            CUDA_CHECK(cuplaMemcpyAsync(
                this->device->getPointer(), /*pointer include X offset*/
                this->host->getBasePointer(),
                hostCurrentSize[0] * sizeof(TYPE),
                cuplaMemcpyHostToDevice,
                this->getCudaStream()));
        }
    };

    template<class TYPE>
    class TaskCopyHostToDevice<TYPE, DIM2> : public TaskCopyHostToDeviceBase<TYPE, DIM2>
    {
    public:
        TaskCopyHostToDevice(HostBuffer<TYPE, DIM2>& src, DeviceBuffer<TYPE, DIM2>& dst)
            : TaskCopyHostToDeviceBase<TYPE, DIM2>(src, dst)
        {
        }

    private:
        virtual void copy(DataSpace<DIM2>& hostCurrentSize)
        {
            CUDA_CHECK(cuplaMemcpy2DAsync(
                this->device->getPointer(),
                this->device->getPitch(), /*this is pitch*/
                this->host->getBasePointer(),
                this->host->getDataSpace()[0] * sizeof(TYPE), /*this is pitch*/
                hostCurrentSize[0] * sizeof(TYPE),
                hostCurrentSize[1],
                cuplaMemcpyHostToDevice,
                this->getCudaStream()));
        }
    };

    template<class TYPE>
    class TaskCopyHostToDevice<TYPE, DIM3> : public TaskCopyHostToDeviceBase<TYPE, DIM3>
    {
    public:
        TaskCopyHostToDevice(HostBuffer<TYPE, DIM3>& src, DeviceBuffer<TYPE, DIM3>& dst)
            : TaskCopyHostToDeviceBase<TYPE, DIM3>(src, dst)
        {
        }

    private:
        virtual void copy(DataSpace<DIM3>& hostCurrentSize)
        {
            cuplaPitchedPtr hostPtr;
            hostPtr.pitch = this->host->getDataSpace()[0] * sizeof(TYPE);
            hostPtr.ptr = this->host->getBasePointer();
            hostPtr.xsize = this->host->getDataSpace()[0] * sizeof(TYPE);
            hostPtr.ysize = this->host->getDataSpace()[1];

            cuplaMemcpy3DParms params;
            params.dstArray = nullptr;
            params.dstPos = make_cuplaPos(
                this->device->getOffset()[0] * sizeof(TYPE),
                this->device->getOffset()[1],
                this->device->getOffset()[2]);
            params.dstPtr = this->device->getCudaPitched();

            params.srcArray = nullptr;
            params.srcPos = make_cuplaPos(0, 0, 0);
            params.srcPtr = hostPtr;

            params.extent
                = make_cuplaExtent(hostCurrentSize[0] * sizeof(TYPE), hostCurrentSize[1], hostCurrentSize[2]);
            params.kind = cuplaMemcpyHostToDevice;

            CUDA_CHECK(cuplaMemcpy3DAsync(&params, this->getCudaStream()));
        }
    };


} // namespace pmacc
