/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "pmacc/eventSystem/tasks/DeviceTask.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"

namespace pmacc
{
    DeviceTask::DeviceTask() : ITask()
    {
        this->setTaskType(ITask::TASK_DEVICE);
    }

    ComputeEventHandle DeviceTask::getComputeEventHandle() const
    {
        PMACC_ASSERT(hasComputeEventHandle);
        return m_alpakaEvent;
    }

    void DeviceTask::setComputeEventHandle(ComputeEventHandle const& alpakaEvent)
    {
        this->hasComputeEventHandle = true;
        this->m_alpakaEvent = alpakaEvent;
    }

    bool DeviceTask::isFinished()
    {
        if(alwaysFinished)
            return true;
        if(hasComputeEventHandle)
        {
            if(m_alpakaEvent.isFinished())
            {
                alwaysFinished = true;
                return true;
            }
        }
        return false;
    }

    Queue* DeviceTask::getComputeDeviceQueue()
    {
        if(stream == nullptr)
            stream = eventSystem::getComputeDeviceQueue(TASK_DEVICE);
        return stream;
    }

    void DeviceTask::setQueue(Queue* newStream)
    {
        PMACC_ASSERT(newStream != nullptr);
        PMACC_ASSERT(stream == nullptr); // it is only allowed to set a stream if no stream is set before
        this->stream = newStream;
    }

    ComputeDeviceQueue DeviceTask::getAlpakaQueue()
    {
        if(stream == nullptr)
            stream = eventSystem::getComputeDeviceQueue(TASK_DEVICE);
        return stream->getAlpakaQueue();
    }

    void DeviceTask::activate()
    {
        m_alpakaEvent = Environment<>::get().EventPool().pop();
        m_alpakaEvent.recordEvent(getAlpakaQueue());
        hasComputeEventHandle = true;
    }

} // namespace pmacc
