/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera
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


#ifndef _TASKSETVALUE_HPP
#define _TASKSETVALUE_HPP

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "memory/buffers/DeviceBuffer.hpp"
#include "dimensions/DataSpace.hpp"
#include "memory/boxes/DataBox.hpp"

#include "eventSystem/EventSystem.hpp"
#include "memory/buffers/DeviceBufferIntern.hpp"
#include "eventSystem/tasks/StreamTask.hpp"
#include "mappings/simulation/EnvironmentController.hpp"

namespace PMacc
{

template <class DataBox>
__global__ void kernelSetValue(DataBox data ,const DataSpace<DIM3> size)
{
    DataSpace<DIM3> idx;

    idx[0] = blockDim.x * (blockIdx.x/size.z()) + threadIdx.x;
    idx[1] = blockDim.y * blockIdx.y;
    idx[2] = blockIdx.x%size.z();

    if (idx.x() >= size.x())
        return;
    data[idx.z()][idx.y()][idx.x()] = *data;
}

template <class DataBox>
__global__ void kernelSetValue(DataBox data, const DataSpace<DIM2> size)
{
    DataSpace<DIM2> idx;

    idx[0] = blockDim.x * blockIdx.x + threadIdx.x;
    idx[1] = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx.x() >= size.x())
        return;

    data[idx.y()][idx.x()] = *data;
}

template <class DataBox>
__global__ void kernelSetValue(DataBox data, const DataSpace<DIM1> size)
{
    size_t idx;

    idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= size.x())
        return;

    data[idx] = *data;
}

template <class DataBox,typename TYPE>
__global__ void kernelSetValue(DataBox data, const TYPE value, const DataSpace<DIM3> size)
{
    DataSpace<DIM3> idx;

    idx[0] = blockDim.x * (blockIdx.x/size.z()) + threadIdx.x;
    idx[1] = blockDim.y * blockIdx.y;
    idx[2] = blockIdx.x%size.z();

    if (idx.x() >= size.x())
        return;
    data[idx.z()][idx.y()][idx.x()] = value;
}

template <class DataBox,typename TYPE>
__global__ void kernelSetValue(DataBox data, const TYPE value, const DataSpace<DIM2> size)
{
    DataSpace<DIM2> idx;

    idx[0] = blockDim.x * blockIdx.x + threadIdx.x;
    idx[1] = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx.x() >= size.x())
        return;

    data[idx.y()][idx.x()] = value;
}

template <class DataBox,typename TYPE>
__global__ void kernelSetValue(DataBox data, const TYPE value, const DataSpace<DIM1> size)
{
    int idx;

    idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= size.x())
        return;

    data[idx] = value;
}


template <class TYPE, unsigned DIM>
class DeviceBufferIntern;

/*Set a value for a GridBuffer on the defice
 * TYPE  = data type (e.g. float, float2)
 * DIM   = dimension of the GridBuffer
 * SMALL = true if TYPE can send via kernel parameter (on cuda TYPE must be smaller than 256 byte)
 */
template <class TYPE, unsigned DIM,bool SMALL>
class TaskSetValue;

template <class TYPE, unsigned DIM>
class TaskSetValue<TYPE,DIM,false> : public StreamTask
{
public:

    TaskSetValue(DeviceBuffer<TYPE, DIM>& dst, const TYPE& value) :
    StreamTask(),
    value(value)
    {
        this->destination = static_cast<DeviceBufferIntern<TYPE, DIM>*> (& dst);
    }

    virtual ~TaskSetValue()
    {
        notify(this->myId, SETVALUE, NULL);
        CUDA_CHECK(cudaFreeHost(valuePointer_host));
    }

    virtual void init()
    {
        setValue();
    }

    bool executeIntern() throw (std::runtime_error)
    {
        return isFinished();
    }

    void event(id_t, EventType, IEventData*)
    {
    }

protected:

    std::string toString()
    {
        return "TaskSetValue";
    }

private:

    void setValue()
    {
        size_t current_size = destination->getCurrentSize();
        const DataSpace<DIM> tmp(destination->getCurrentDataSpace(current_size));
        dim3 gridSize = tmp;

        gridSize.x = (gridSize.x+255) / 256; //round up without ceil
        gridSize.x *= gridSize.z;
        gridSize.z = 1;

        CUDA_CHECK(cudaMallocHost(&valuePointer_host, sizeof (TYPE)));
        *valuePointer_host = value; //copy value to new place

        CUDA_CHECK(cudaMemcpyAsync(
                                   destination->getPointer(), valuePointer_host, sizeof (TYPE),
                                   cudaMemcpyHostToDevice, this->getCudaStream()));
        kernelSetValue << <gridSize, 256, 0, this->getCudaStream() >> >
                (destination->getDataBox(), tmp);

        this->activate();
    }

    DeviceBufferIntern<TYPE, DIM> *destination;
    //TYPE *valuePointer_dev;
    TYPE *valuePointer_host;
    TYPE value;
};

template <class TYPE, unsigned DIM>
class TaskSetValue<TYPE,DIM,true> : public StreamTask
{
public:

    TaskSetValue(DeviceBuffer<TYPE, DIM>& dst, const TYPE &value) :
    StreamTask(),
    value(value)
    {
        this->destination = static_cast<DeviceBufferIntern<TYPE, DIM>*> (& dst);
    }

    virtual ~TaskSetValue()
    {
        notify(this->myId, SETVALUE, NULL);

    }

    virtual void init()
    {

        setValue();

    }

    bool executeIntern() throw (std::runtime_error)
    {
        return isFinished();
    }

    void event(id_t, EventType, IEventData*)
    {
    }

protected:

    std::string toString()
    {
        return "TaskSetValueSmall";
    }

private:

    void setValue()
    {

        size_t current_size = destination->getCurrentSize();
        DataSpace<DIM> tmp = destination->getCurrentDataSpace(current_size);
        dim3 gridSize = tmp;

        gridSize.x = (gridSize.x+255) / 256; //round up without ceil
        gridSize.x *= gridSize.z;
        gridSize.z = 1;

        kernelSetValue << <gridSize, 256, 0, this->getCudaStream() >> >
                (destination->getDataBox(), value,tmp);

        this->activate();
    }

    DeviceBufferIntern<TYPE, DIM> *destination;
    TYPE value;
};

} //namespace PMacc


#endif	/* _TASKSETVALUE_HPP */
