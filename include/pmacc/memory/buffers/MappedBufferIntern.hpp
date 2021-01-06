/* Copyright 2014-2021 Rene Widera, Axel Huebl, Benjamin Worpitz,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/assert.hpp"

namespace pmacc
{
    /** Implementation of the DeviceBuffer interface for cuda mapped memory
     *
     * For all pmacc tasks and functions this buffer looks like native device buffer
     * but in real it is stored in host memory.
     */
    template<class TYPE, unsigned DIM>
    class MappedBufferIntern : public DeviceBuffer<TYPE, DIM>
    {
        /** IMPORTANT: if someone implements that a MappedBufferIntern can points to an other
         * mapped buffer then `getDataSpace()` in `getHostDataBox()` and `getDeviceDataBox`
         * must be changed to `getPhysicalMemorySize`
         */
    public:
        typedef typename DeviceBuffer<TYPE, DIM>::DataBoxType DataBoxType;

        /** constructor
         *
         * @param size extent for each dimension (in elements)
         */
        MappedBufferIntern(DataSpace<DIM> size)
            : DeviceBuffer<TYPE, DIM>(size, size)
            , pointer(nullptr)
            , ownPointer(true)
        {
#if(PMACC_CUDA_ENABLED == 1)
            CUDA_CHECK((
                cuplaError_t) cudaHostAlloc(&pointer, size.productOfComponents() * sizeof(TYPE), cudaHostAllocMapped));
#else
            pointer = new TYPE[size.productOfComponents()];
#endif
            reset(false);
        }

        /**
         * destructor
         */
        virtual ~MappedBufferIntern()
        {
            __startOperation(ITask::TASK_DEVICE);
            __startOperation(ITask::TASK_HOST);

            if(pointer && ownPointer)
            {
#if(PMACC_CUDA_ENABLED == 1)
/* cupla 0.2.0 does not support the function cudaHostAlloc to create mapped memory.
 * Therefore we need to call the native CUDA function cudaFreeHost to free memory.
 * Due to the renaming of cuda functions with cupla via macros we need to remove
 * the renaming to get access to the native cuda function.
 * @todo this is a workaround please fix me. We need to investigate if
 * it is possible to have mapped/unified memory in alpaka.
 *
 * corresponding alpaka issues:
 *   https://github.com/ComputationalRadiationPhysics/alpaka/issues/296
 *   https://github.com/ComputationalRadiationPhysics/alpaka/issues/612
 */
#    undef cudaFreeHost
                CUDA_CHECK((cuplaError_t) cudaFreeHost(pointer));
// re-introduce the cupla macro
#    define cudaFreeHost(...) cuplaFreeHost(__VA_ARGS__)
#else
                __deleteArray(pointer);
#endif
            }
        }

        /*! Get unchanged device pointer of memory
         * @return device pointer to memory
         */
        TYPE* getBasePointer()
        {
            __startOperation(ITask::TASK_HOST);
            return (TYPE*) this->getCudaPitched().ptr;
        }

        /*! Get device pointer of memory
         *
         * This pointer is shifted by the offset, if this buffer points to other
         * existing buffer
         *
         * @return device pointer to memory
         */
        TYPE* getPointer()
        {
            __startOperation(ITask::TASK_HOST);
            return (TYPE*) this->getCudaPitched().ptr;
        }

        void copyFrom(HostBuffer<TYPE, DIM>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopyHostToDevice(other, *this);
        }

        void copyFrom(DeviceBuffer<TYPE, DIM>& other)
        {
            PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
            Environment<>::get().Factory().createTaskCopyDeviceToDevice(other, *this);
        }

        void reset(bool preserveData = true)
        {
            __startOperation(ITask::TASK_HOST);
            this->setCurrentSize(this->getDataSpace().productOfComponents());
            if(!preserveData)
                memset(pointer, 0, this->getDataSpace().productOfComponents() * sizeof(TYPE));
        }

        void setValue(const TYPE& value)
        {
            __startOperation(ITask::TASK_HOST);
            size_t current_size = this->getCurrentSize();
            for(size_t i = 0; i < current_size; i++)
            {
                pointer[i] = value;
            }
        }

        bool hasCurrentSizeOnDevice() const
        {
            return false;
        }

        virtual size_t* getCurrentSizeHostSidePointer()
        {
            return this->current_size;
        }

        size_t* getCurrentSizeOnDevicePointer()
        {
            return nullptr;
        }

        DataSpace<DIM> getOffset() const
        {
            return DataSpace<DIM>();
        }

        void setCurrentSize(const size_t size)
        {
            Buffer<TYPE, DIM>::setCurrentSize(size);
        }

        const cuplaPitchedPtr getCudaPitched() const
        {
            __startOperation(ITask::TASK_DEVICE);
            TYPE* dPointer;
            cuplaHostGetDevicePointer(&dPointer, pointer, 0);

            /* on 1D memory we have no size for y, therefore we set y to 1 to
             * get a valid cuplaPitchedPtr
             */
            int size_y = 1;
            if(DIM > DIM1)
                size_y = this->data_space[1];

            return make_cuplaPitchedPtr(dPointer, this->data_space.x() * sizeof(TYPE), this->data_space.x(), size_y);
        }

        size_t getPitch() const
        {
            return this->data_space.x() * sizeof(TYPE);
        }

        DataBoxType getHostDataBox()
        {
            __startOperation(ITask::TASK_HOST);
            return DataBoxType(PitchedBox<TYPE, DIM>(
                pointer,
                DataSpace<DIM>(),
                this->data_space,
                this->data_space[0] * sizeof(TYPE)));
        }

        DataBoxType getDataBox()
        {
            __startOperation(ITask::TASK_DEVICE);
            TYPE* dPointer;
            cuplaHostGetDevicePointer(&dPointer, pointer, 0);
            return DataBoxType(PitchedBox<TYPE, DIM>(
                dPointer,
                DataSpace<DIM>(),
                this->data_space,
                this->data_space[0] * sizeof(TYPE)));
        }

    private:
        TYPE* pointer;
        bool ownPointer;
    };

} // namespace pmacc
