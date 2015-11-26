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

#include "memory/buffers/HostBuffer.hpp"
#include "eventSystem/tasks/Factory.hpp"
#include "eventSystem/EventSystem.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"

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

    typedef typename HostBuffer<TYPE, DIM>::DataBoxType DataBoxType;

    /** constructor
     *
     * @param size extent for each dimension (in elements)
     */
    HostBufferIntern(DataSpace<DIM> size) :
        HostBuffer<TYPE, DIM>(size, size),
        pointer(NULL),
        basePointer(NULL),
        ownsPointer(true)
    {
        CUDA_CHECK(cudaMallocHost(&pointer, size.productOfComponents() * sizeof (TYPE)));
        basePointer = pointer;
        reset(false);
    }

    HostBufferIntern(HostBufferIntern& source, DataSpace<DIM> size, DataSpace<DIM> offset=DataSpace<DIM>()) :
        HostBuffer<TYPE, DIM>(size, source.getPhysicalMemorySize()),
        pointer(NULL),
        basePointer(NULL),
        ownsPointer(false)
    {
        pointer=&(source.getDataBox()(offset));/*fix me, this is a bad way*/
        basePointer = source.getPointer();
        reset(true);
    }

    /**
     * destructor
     */
    virtual ~HostBufferIntern()
    {
        __startOperation(ITask::TASK_HOST);

        if (this->getPointer() && ownsPointer)
        {
            CUDA_CHECK(cudaFreeHost(this->getPointer()));
        }
    }

    /*! Get pointer of memory
     * @return pointer to memory
     */
    TYPE* getBasePointer()
    {
        __startOperation(ITask::TASK_HOST);
        return basePointer;
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
        {
            /* if it is a pointer out of other memory we can not assume that
             * that the physical memory is contiguous
             */
            if(ownsPointer)
                memset(this->getPointer(), 0, this->getDataSpace().productOfComponents() * sizeof (TYPE));
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
        return DataBoxType(PitchedBox<TYPE, DIM > (this->getPointer(), DataSpace<DIM > (),
                                                   this->getPhysicalMemorySize(), this->getPhysicalMemorySize()[0] * sizeof (TYPE)));
    }

private:
    TYPE* pointer;
    TYPE* basePointer;
    bool ownsPointer;
};

}
