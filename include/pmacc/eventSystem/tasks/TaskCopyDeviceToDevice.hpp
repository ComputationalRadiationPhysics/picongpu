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
#include "pmacc/types.hpp"


namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class TaskCopyDeviceToDeviceBase : public StreamTask
    {
    public:
        TaskCopyDeviceToDeviceBase(DeviceBuffer<TYPE, DIM>& src, DeviceBuffer<TYPE, DIM>& dst) : StreamTask()
        {
            this->source = &src;
            this->destination = &dst;
        }

        virtual ~TaskCopyDeviceToDeviceBase()
        {
            notify(this->myId, COPYDEVICE2DEVICE, nullptr);
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
            size_t current_size = source->getCurrentSize();
            destination->setCurrentSize(current_size);
            DataSpace<DIM> devCurrentSize = source->getCurrentDataSpace(current_size);
            if(source->is1D() && destination->is1D())
                fastCopy(source->getPointer(), destination->getPointer(), devCurrentSize.productOfComponents());
            else
                copy(devCurrentSize);

            this->activate();
        }

        std::string toString()
        {
            return "TaskCopyDeviceToDevice";
        }

    protected:
        virtual void copy(DataSpace<DIM>& devCurrentSize) = 0;

        void fastCopy(TYPE* src, TYPE* dst, size_t size)
        {
            CUDA_CHECK(
                cuplaMemcpyAsync(dst, src, size * sizeof(TYPE), cuplaMemcpyDeviceToDevice, this->getCudaStream()));
        }

        DeviceBuffer<TYPE, DIM>* source;
        DeviceBuffer<TYPE, DIM>* destination;
    };


    template<class TYPE, unsigned DIM>
    class TaskCopyDeviceToDevice;

    template<class TYPE>
    class TaskCopyDeviceToDevice<TYPE, DIM1> : public TaskCopyDeviceToDeviceBase<TYPE, DIM1>
    {
    public:
        TaskCopyDeviceToDevice(DeviceBuffer<TYPE, DIM1>& src, DeviceBuffer<TYPE, DIM1>& dst)
            : TaskCopyDeviceToDeviceBase<TYPE, DIM1>(src, dst)
        {
        }

    private:
        virtual void copy(DataSpace<DIM1>& devCurrentSize)
        {
            CUDA_CHECK(cuplaMemcpyAsync(
                this->destination->getPointer(),
                this->source->getPointer(),
                devCurrentSize[0] * sizeof(TYPE),
                cuplaMemcpyDeviceToDevice,
                this->getCudaStream()));
        }
    };

    template<class TYPE>
    class TaskCopyDeviceToDevice<TYPE, DIM2> : public TaskCopyDeviceToDeviceBase<TYPE, DIM2>
    {
    public:
        TaskCopyDeviceToDevice(DeviceBuffer<TYPE, DIM2>& src, DeviceBuffer<TYPE, DIM2>& dst)
            : TaskCopyDeviceToDeviceBase<TYPE, DIM2>(src, dst)
        {
        }

    private:
        virtual void copy(DataSpace<DIM2>& devCurrentSize)
        {
            CUDA_CHECK(cuplaMemcpy2DAsync(
                this->destination->getPointer(),
                this->destination->getPitch(),
                this->source->getPointer(),
                this->source->getPitch(),
                devCurrentSize[0] * sizeof(TYPE),
                devCurrentSize[1],
                cuplaMemcpyDeviceToDevice,
                this->getCudaStream()));
        }
    };

    template<class TYPE>
    class TaskCopyDeviceToDevice<TYPE, DIM3> : public TaskCopyDeviceToDeviceBase<TYPE, DIM3>
    {
    public:
        TaskCopyDeviceToDevice(DeviceBuffer<TYPE, DIM3>& src, DeviceBuffer<TYPE, DIM3>& dst)
            : TaskCopyDeviceToDeviceBase<TYPE, DIM3>(src, dst)
        {
        }

    private:
        virtual void copy(DataSpace<DIM3>& devCurrentSize)
        {
            cuplaMemcpy3DParms params;
            params.srcArray = nullptr;
            params.srcPos = make_cuplaPos(
                this->source->getOffset()[0] * sizeof(TYPE),
                this->source->getOffset()[1],
                this->source->getOffset()[2]);
            params.srcPtr = this->source->getCudaPitched();

            params.dstArray = nullptr;
            params.dstPos = make_cuplaPos(
                this->destination->getOffset()[0] * sizeof(TYPE),
                this->destination->getOffset()[1],
                this->destination->getOffset()[2]);
            ;
            params.dstPtr = this->destination->getCudaPitched();

            params.extent = make_cuplaExtent(devCurrentSize[0] * sizeof(TYPE), devCurrentSize[1], devCurrentSize[2]);
            params.kind = cuplaMemcpyDeviceToDevice;
            CUDA_CHECK(cuplaMemcpy3DAsync(&params, this->getCudaStream()));
        }
    };

} // namespace pmacc
