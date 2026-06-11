/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/tasks/DeviceTask.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    template<typename T_DeviceBuffer>
    class TaskGetCurrentSizeFromDevice : public DeviceTask
    {
    public:
        TaskGetCurrentSizeFromDevice(T_DeviceBuffer& buff) : DeviceTask(), buffer(&buff)
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
