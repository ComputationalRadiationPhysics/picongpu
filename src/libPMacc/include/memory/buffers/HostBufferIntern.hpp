/**
 * Copyright 2013-2015 Rene Widera, Benjamin Worpitz,
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

#include "memory/buffers/Buffer.hpp"
#include "eventSystem/tasks/Factory.hpp"
#include "eventSystem/EventSystem.hpp"

#include <cassert>

namespace PMacc
{

/**
 * Internal implementation of the HostBuffer interface.
 */
template <class TYPE, unsigned DIM>
class HostBufferIntern : public HostBuffer<TYPE, DIM>
{
public:

    typedef typename DeviceBuffer<TYPE, DIM>::DataBoxType DataBoxType;

    /**
     * constructor
     * @param dataSpace DataSpace describing the size of the HostBufferIntern to be created
     */
    HostBufferIntern(DataSpace<DIM> dataSpace) :
    HostBuffer<TYPE, DIM>(dataSpace),
    pointer(NULL),ownPointer(true)
    {
        CUDA_CHECK(cudaMallocHost(&pointer, dataSpace.productOfComponents() * sizeof (TYPE)));
        reset(false);
    }

    HostBufferIntern(HostBufferIntern& source, DataSpace<DIM> dataSpace, DataSpace<DIM> offset=DataSpace<DIM>()) :
    HostBuffer<TYPE, DIM>(dataSpace),
    pointer(NULL),ownPointer(false)
    {
        pointer=&(source.getDataBox()(offset));/*fix me, this is a bad way*/
        reset(true);
    }

    /**
     * destructor
     */
    virtual ~HostBufferIntern()
    {
        __startOperation(ITask::TASK_HOST);

        if (pointer && ownPointer)
        {
            CUDA_CHECK(cudaFreeHost(pointer));
        }
    }

    /*! Get pointer of memory
     * @return pointer to memory
     */
    TYPE* getBasePointer()
    {
        __startOperation(ITask::TASK_HOST);
        return pointer;
    }

    TYPE* getPointer()
    {
        __startOperation(ITask::TASK_HOST);
        return pointer;
    }

    void copyFrom(DeviceBuffer<TYPE, DIM>& other)
    {
        assert(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        Environment<>::get().Factory().createTaskCopyDeviceToHost(other, *this);
    }

    void reset(bool preserveData = true)
    {
        __startOperation(ITask::TASK_HOST);
        this->setCurrentSize(this->getDataSpace().productOfComponents());
        if (!preserveData)
            memset(pointer, 0, this->getDataSpace().productOfComponents() * sizeof (TYPE));
    }

    void setValue(const TYPE& value)
    {
        __startOperation(ITask::TASK_HOST);
        size_t current_size = this->getCurrentSize();
        for (size_t i = 0; i < current_size; i++)
        {
            pointer[i] = value;
        }
    }

    DataBoxType getDataBox()
    {
        __startOperation(ITask::TASK_HOST);
        return DataBoxType(PitchedBox<TYPE, DIM > (pointer, DataSpace<DIM > (),
                                                   this->data_space, this->data_space[0] * sizeof (TYPE)));
    }

private:
    TYPE* pointer;
    bool ownPointer;
};

}
