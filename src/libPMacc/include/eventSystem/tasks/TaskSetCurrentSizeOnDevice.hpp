/**
 * Copyright 2013-2015 Felix Schmitt, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "eventSystem/EventSystem.hpp"
#include "eventSystem/streams/EventStream.hpp"
#include "eventSystem/tasks/StreamTask.hpp"
#include "eventSystem/events/kernelEvents.hpp"
#include "dimensions/DataSpace.hpp"

#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void kernelSetValueOnDeviceMemory(size_t* pointer, const size_t size)
{
    *pointer = size;
}

namespace PMacc
{

template <class TYPE, unsigned DIM>
class DeviceBuffer;

template <class TYPE, unsigned DIM>
class TaskSetCurrentSizeOnDevice : public StreamTask
{
public:

    TaskSetCurrentSizeOnDevice(DeviceBuffer<TYPE, DIM>& dst, size_t size) :
    StreamTask(),
    size(size)
    {
        this->destination = & dst;
    }

    virtual ~TaskSetCurrentSizeOnDevice()
    {
        notify(this->myId, SETVALUE, NULL);
    }

    virtual void init()
    {
        setSize();
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
        return "TaskSetCurrentSizeOnDevice";
    }

private:

    void setSize()
    {
        kernelSetValueOnDeviceMemory
            << < 1, 1, 0, this->getCudaStream() >> >
            (destination->getCurrentSizeOnDevicePointer(), size);

        activate();
    }

    DeviceBuffer<TYPE, DIM> *destination;
    const size_t size;
};

} //namespace PMacc

