/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Benjamin Worpitz,
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

#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/eventSystem/streams/EventStream.hpp"
#include "pmacc/eventSystem/tasks/StreamTask.hpp"
#include "pmacc/eventSystem/events/kernelEvents.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/nvidia/gpuEntryFunction.hpp"


namespace pmacc
{
    struct KernelSetValueOnDeviceMemory
    {
        template<typename T_Acc>
        DINLINE void operator()(const T_Acc&, size_t* pointer, const size_t size) const
        {
            *pointer = size;
        }
    };

    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class TaskSetCurrentSizeOnDevice : public StreamTask
    {
    public:
        TaskSetCurrentSizeOnDevice(DeviceBuffer<TYPE, DIM>& dst, size_t size) : StreamTask(), size(size)
        {
            this->destination = &dst;
        }

        virtual ~TaskSetCurrentSizeOnDevice()
        {
            notify(this->myId, SETVALUE, nullptr);
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
            auto sizePtr = destination->getCurrentSizeOnDevicePointer();
            CUPLA_KERNEL(KernelSetValueOnDeviceMemory)(1, 1, 0, this->getCudaStream())(sizePtr, size);

            activate();
        }

        DeviceBuffer<TYPE, DIM>* destination;
        const size_t size;
    };

} // namespace pmacc
