/**
 * Copyright 2013-2016 Rene Widera, Benjamin Worpitz,
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

#include "memory/buffers/HostBuffer.hpp"
#include "eventSystem/tasks/Factory.hpp"
#include "eventSystem/EventSystem.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"
#include "assert.hpp"

namespace PMacc
{

/**
 * Internal implementation of the HostBuffer interface.
 */
template <class TYPE, unsigned DIM>
class HostBufferIntern : public HostBuffer<TYPE, DIM>
{
public:

    typedef typename HostBuffer<TYPE, DIM>::DataBoxType DataBoxType;

    /** constructor
     *
     * @param size extent for each dimension (in elements)
     */
    HostBufferIntern(DataSpace<DIM> size) :
    HostBuffer<TYPE, DIM>(size, size),
    pointer(NULL),ownPointer(true)
    {
        CUDA_CHECK(cudaMallocHost(&pointer, size.productOfComponents() * sizeof (TYPE)));
        reset(false);
    }

    HostBufferIntern(HostBufferIntern& source, DataSpace<DIM> size, DataSpace<DIM> offset=DataSpace<DIM>()) :
    HostBuffer<TYPE, DIM>(size, source.getPhysicalMemorySize()),
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
        PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        Environment<>::get().Factory().createTaskCopyDeviceToHost(other, *this);
    }

    void reset(bool preserveData = true)
    {
        __startOperation(ITask::TASK_HOST);
        this->setCurrentSize(this->getDataSpace().productOfComponents());
        if (!preserveData)
        {
            /* if it is a pointer out of other memory we can not assume that
             * that the physical memory is contiguous
             */
            if(ownPointer)
                memset(pointer, 0, this->getDataSpace().productOfComponents() * sizeof (TYPE));
            else
            {
                TYPE value;
                /* using `uint8_t` for byte-wise looping through tmp var value of `TYPE` */
                uint8_t* valuePtr = (uint8_t*)&value;
                for( size_t b = 0; b < sizeof(TYPE); ++b)
                {
                    valuePtr[b] = static_cast<uint8_t>(0);
                }
                /* set value with zero-ed `TYPE` */
                setValue(value);
            }
        }
    }

    void setValue(const TYPE& value)
    {
        __startOperation(ITask::TASK_HOST);
        size_t current_size = this->getCurrentSize();
        PMACC_AUTO(memBox,getDataBox());
        typedef DataBoxDim1Access<DataBoxType > D1Box;
        D1Box d1Box(memBox, this->getDataSpace());
        #pragma omp parallel for
        for (size_t i = 0; i < current_size; i++)
        {
            d1Box[i] = value;
        }
    }

    DataBoxType getDataBox()
    {
        __startOperation(ITask::TASK_HOST);
        return DataBoxType(PitchedBox<TYPE, DIM > (pointer, DataSpace<DIM > (),
                                                   this->getPhysicalMemorySize(), this->getPhysicalMemorySize()[0] * sizeof (TYPE)));
    }

private:
    TYPE* pointer;
    bool ownPointer;
};

}
