/**
 * Copyright 2013 René Widera
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


#ifndef _HOSTBUFFERINTERN_HPP
#define	_HOSTBUFFERINTERN_HPP

#include <cassert>

#include "memory/buffers/Buffer.hpp"
#include "eventSystem/EventSystem.hpp"

#include "eventSystem/tasks/Factory.hpp"

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
    HostBufferIntern(DataSpace<DIM> dataSpace) throw (std::bad_alloc) :
    HostBuffer<TYPE, DIM>(dataSpace),
    pointer(NULL),ownPointer(true)
    {
        CUDA_CHECK(cudaMallocHost(&pointer, dataSpace.getElementCount() * sizeof (TYPE)));
        reset(false);
    }

    HostBufferIntern(HostBufferIntern& source, DataSpace<DIM> dataSpace, DataSpace<DIM> offset=DataSpace<DIM>())  :
    HostBuffer<TYPE, DIM>(dataSpace),
    pointer(NULL),ownPointer(false)
    {
        pointer=&(source.getDataBox()(offset));/*fix me, this is a bad way*/
        reset(true);
    }

    /**
     * destructor
     */
    virtual ~HostBufferIntern() throw (std::runtime_error)
    {
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
        Factory::getInstance().createTaskCopyDeviceToHost(other, *this);
    }

    void reset(bool preserveData = true)
    {
        __startOperation(ITask::TASK_HOST);
        this->setCurrentSize(this->getDataSpace().getElementCount());
        if (!preserveData)
            memset(pointer, 0, this->getDataSpace().getElementCount() * sizeof (TYPE));
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

#endif	/* _HOSTBUFFERINTERN_HPP */

