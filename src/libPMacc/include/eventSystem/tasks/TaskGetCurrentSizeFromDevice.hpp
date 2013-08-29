/**
 * Copyright 2013 Felix Schmitt, René Widera
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
 
#ifndef _TASKGETCURRENTSIZEFROMDEVICE_HPP
#define _TASKGETCURRENTSIZEFROMDEVICE_HPP

#include <cuda_runtime_api.h>
#include <cuda.h>


#include "dimensions/DataSpace.hpp"
#include "types.h"

#include "eventSystem/EventSystem.hpp"
#include "eventSystem/streams/EventStream.hpp"
#include "eventSystem/tasks/StreamTask.hpp"

namespace PMacc
{


template <class TYPE, unsigned DIM>
class DeviceBufferIntern;

template <class TYPE, unsigned DIM>
class TaskGetCurrentSizeFromDevice : public StreamTask
{
public:

    TaskGetCurrentSizeFromDevice(DeviceBuffer<TYPE,DIM>& buffer):
    StreamTask()
    {
        this->buffer = (DeviceBufferIntern<TYPE, DIM>*) & buffer;
    }

    virtual ~TaskGetCurrentSizeFromDevice()
    {
        notify(this->myId,GETVALUE, NULL);
    }

    bool executeIntern() throw(std::runtime_error)
    {   
        return isFinished();
    }

    void event(id_t, EventType, IEventData*)
    {
    }

    virtual void init()
    {
        CUDA_CHECK(cudaMemcpyAsync((void*) buffer->getCurrentSizeHostSidePointer(),
                                   buffer->getCurrentSizeOnDevicePointer(),
                                   sizeof (size_t),
                                   cudaMemcpyDeviceToHost,
                                   this->getCudaStream()));
        this->activate();
    }

    virtual std::string toString()
    {
        return "TaskGetCurrentSizeFromDevice";
    }

private:

    DeviceBufferIntern<TYPE, DIM> *buffer;
};

} //namespace PMacc


#endif	/* _TASKGETCURRENTSIZEFROMDEVICE_HPP */

