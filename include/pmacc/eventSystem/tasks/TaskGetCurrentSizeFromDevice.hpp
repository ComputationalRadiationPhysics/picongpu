/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Benjamin Worpitz,
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/tasks/StreamTask.hpp"
#include "pmacc/types.hpp"


namespace pmacc
{
    template<typename T_DeviceBuffer>
    class TaskGetCurrentSizeFromDevice : public StreamTask
    {
    public:
        TaskGetCurrentSizeFromDevice(T_DeviceBuffer& buff) : StreamTask(), buffer(&buff)
        {
        }

        ~TaskGetCurrentSizeFromDevice() override
        {
            notify(this->myId, GETVALUE, nullptr);
        }

        bool executeIntern() override
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        void init() override
        {
            auto queue = this->getAlpakaQueue();
            alpaka::memcpy(
                queue,
                buffer->sizeHostSideBuffer(),
                buffer->sizeDeviceSideBuffer(),
                MemSpace<DIM1>(1).toAlpakaMemVec());
            this->activate();
        }

        std::string toString() override
        {
            return "TaskGetCurrentSizeFromDevice";
        }

    private:
        T_DeviceBuffer* buffer;
    };

} // namespace pmacc
