/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/memory/boxes/DataBoxDim1Access.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/assert.hpp"

namespace pmacc
{
    /**
     * Internal implementation of the HostBuffer interface.
     */
    template<class TYPE, unsigned DIM>
    class HostBufferIntern : public HostBuffer<TYPE, DIM>
    {
    public:
        typedef typename HostBuffer<TYPE, DIM>::DataBoxType DataBoxType;

        /** constructor
         *
         * @param size extent for each dimension (in elements)
         */
        HostBufferIntern(DataSpace<DIM> size) : HostBuffer<TYPE, DIM>(size, size), pointer(nullptr), ownPointer(true)
        {
            CUDA_CHECK(cuplaMallocHost((void**) &pointer, size.productOfComponents() * sizeof(TYPE)));
            reset(false);
        }

        HostBufferIntern(HostBufferIntern& source, DataSpace<DIM> size, DataSpace<DIM> offset = DataSpace<DIM>())
            : HostBuffer<TYPE, DIM>(size, source.getPhysicalMemorySize())
            , pointer(nullptr)
            , ownPointer(false)
        {
            pointer = &(source.getDataBox()(offset)); /*fix me, this is a bad way*/
            reset(true);
        }

        /**
         * destructor
         */
        virtual ~HostBufferIntern()
        {
            __startOperation(ITask::TASK_HOST);

            if(pointer && ownPointer)
            {
                CUDA_CHECK_NO_EXCEPT(cuplaFreeHost(pointer));
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
            if(!preserveData)
            {
                /* if it is a pointer out of other memory we can not assume that
                 * that the physical memory is contiguous
                 */
                if(ownPointer)
                    memset(
                        reinterpret_cast<void*>(pointer),
                        0,
                        this->getDataSpace().productOfComponents() * sizeof(TYPE));
                else
                {
                    // Using Array is a workaround for types without default constructor
                    memory::Array<TYPE, 1> tmp;
                    memset(reinterpret_cast<void*>(tmp.data()), 0, sizeof(tmp));
                    // use first element to avoid issue because Array is aligned (sizeof can be larger than component
                    // type)
                    setValue(tmp[0]);
                }
            }
        }

        void setValue(const TYPE& value)
        {
            __startOperation(ITask::TASK_HOST);
            int64_t current_size = static_cast<int64_t>(this->getCurrentSize());
            auto memBox = getDataBox();
            typedef DataBoxDim1Access<DataBoxType> D1Box;
            D1Box d1Box(memBox, this->getDataSpace());
#pragma omp parallel for
            for(int64_t i = 0; i < current_size; i++)
            {
                d1Box[i] = value;
            }
        }

        DataBoxType getDataBox()
        {
            __startOperation(ITask::TASK_HOST);
            return DataBoxType(PitchedBox<TYPE, DIM>(
                pointer,
                DataSpace<DIM>(),
                this->getPhysicalMemorySize(),
                this->getPhysicalMemorySize()[0] * sizeof(TYPE)));
        }

    private:
        TYPE* pointer;
        bool ownPointer;
    };

} // namespace pmacc
