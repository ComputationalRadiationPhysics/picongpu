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
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/types.hpp"


namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class TaskGetCurrentSizeFromDevice : public StreamTask
    {
    public:
        TaskGetCurrentSizeFromDevice(DeviceBuffer<TYPE, DIM>& buffer) : StreamTask()
        {
            this->buffer = &buffer;
        }

        virtual ~TaskGetCurrentSizeFromDevice()
        {
            notify(this->myId, GETVALUE, nullptr);
        }

        bool executeIntern()
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*)
        {
        }

        virtual void init()
        {
            CUDA_CHECK(cuplaMemcpyAsync(
                (void*) buffer->getCurrentSizeHostSidePointer(),
                buffer->getCurrentSizeOnDevicePointer(),
                sizeof(size_t),
                cuplaMemcpyDeviceToHost,
                this->getCudaStream()));
            this->activate();
        }

        virtual std::string toString()
        {
            return "TaskGetCurrentSizeFromDevice";
        }

    private:
        DeviceBuffer<TYPE, DIM>* buffer;
    };

} // namespace pmacc
