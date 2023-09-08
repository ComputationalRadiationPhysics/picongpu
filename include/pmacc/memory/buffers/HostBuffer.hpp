/* Copyright 2013-2023 Rene Widera, Benjamin Worpitz,
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

#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/eventSystem.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/memory/boxes/DataBoxDim1Access.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"

namespace pmacc
{
    class EventTask;

    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class Buffer;

    /** DIM-dimensional Buffer of type TYPE on the host
     *
     * @tparam TYPE datatype for buffer data
     * @tparam DIM dimension of the buffer
     */
    template<class TYPE, unsigned DIM>
    class HostBuffer : public Buffer<TYPE, DIM>
    {
    public:
        using DataBoxType = typename Buffer<TYPE, DIM>::DataBoxType;

        /** constructor
         *
         * @param size extent for each dimension (in elements)
         */
        HostBuffer(DataSpace<DIM> size) : Buffer<TYPE, DIM>(size, size), pointer(nullptr), ownPointer(true)
        {
            CUDA_CHECK(cuplaMallocHost((void**) &pointer, size.productOfComponents() * sizeof(TYPE)));
            reset(false);
        }

        HostBuffer(HostBuffer& source, DataSpace<DIM> size, DataSpace<DIM> offset = DataSpace<DIM>())
            : Buffer<TYPE, DIM>(size, source.getPhysicalMemorySize())
            , pointer(nullptr)
            , ownPointer(false)
        {
            pointer = &(source.getDataBox()(offset)); /*fix me, this is a bad way*/
            reset(true);
        }

        /**
         * destructor
         */
        ~HostBuffer() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);

            if(pointer && ownPointer)
            {
                CUDA_CHECK_NO_EXCEPT(cuplaFreeHost(pointer));
            }
        }

        /**
         * Returns the current size pointer.
         *
         * @return pointer to current size
         */
        size_t* getCurrentSizePointer()
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return this->current_size;
        }

        /*! Get pointer of memory
         * @return pointer to memory
         */
        TYPE* getBasePointer() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return pointer;
        }

        TYPE* getPointer() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return pointer;
        }

        /**
         * Copies the data from the given DeviceBuffer to this HostBuffer.
         *
         * @param other DeviceBuffer to copy data from
         */
        void copyFrom(DeviceBuffer<TYPE, DIM>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopyDeviceToHost(other, *this);
        }

        void reset(bool preserveData = true) override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
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

        void setValue(const TYPE& value) override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            auto current_size = static_cast<int64_t>(this->getCurrentSize());
            auto memBox = getDataBox();
            using D1Box = DataBoxDim1Access<DataBoxType>;
            D1Box d1Box(memBox, this->getDataSpace());
#pragma omp parallel for
            for(int64_t i = 0; i < current_size; i++)
            {
                d1Box[i] = value;
            }
        }

        DataBoxType getDataBox() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return DataBoxType(PitchedBox<TYPE, DIM>(
                pointer,
                this->getPhysicalMemorySize(),
                this->getPhysicalMemorySize()[0] * sizeof(TYPE)));
        }

    private:
        TYPE* pointer;
        bool ownPointer;
    };

} // namespace pmacc
