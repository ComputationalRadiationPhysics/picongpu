/* Copyright 2013-2023 Rene Widera, Benjamin Worpitz
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


#include "pmacc/eventSystem/tasks/StreamTask.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"

namespace pmacc
{
    StreamTask::StreamTask() : ITask()
    {
        this->setTaskType(ITask::TASK_DEVICE);
    }

    CudaEventHandle StreamTask::getCudaEventHandle() const
    {
        PMACC_ASSERT(hasCudaEventHandle);
        return m_alpakaEvent;
    }

    void StreamTask::setCudaEventHandle(const CudaEventHandle& alpakaEvent)
    {
        this->hasCudaEventHandle = true;
        this->m_alpakaEvent = alpakaEvent;
    }

    bool StreamTask::isFinished()
    {
        if(alwaysFinished)
            return true;
        if(hasCudaEventHandle)
        {
            if(m_alpakaEvent.isFinished())
            {
                alwaysFinished = true;
                return true;
            }
        }
        return false;
    }

    Queue* StreamTask::getComputeDeviceQueue()
    {
        if(stream == nullptr)
            stream = eventSystem::getComputeDeviceQueue(TASK_DEVICE);
        return stream;
    }

    void StreamTask::setQueue(Queue* newStream)
    {
        PMACC_ASSERT(newStream != nullptr);
        PMACC_ASSERT(stream == nullptr); // it is only allowed to set a stream if no stream is set before
        this->stream = newStream;
    }

    AccStream StreamTask::getAlpakaQueue()
    {
        if(stream == nullptr)
            stream = eventSystem::getComputeDeviceQueue(TASK_DEVICE);
        return stream->getAlpakaQueue();
    }

    void StreamTask::activate()
    {
        m_alpakaEvent = Environment<>::get().EventPool().pop();
        m_alpakaEvent.recordEvent(getAlpakaQueue());
        hasCudaEventHandle = true;
    }

} // namespace pmacc
