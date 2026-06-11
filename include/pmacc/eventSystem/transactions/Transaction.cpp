/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pmacc/eventSystem/transactions/Transaction.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/events/EventTask.hpp"
#include "pmacc/eventSystem/queues/QueueController.hpp"
#include "pmacc/eventSystem/tasks/DeviceTask.hpp"

namespace pmacc
{
    Transaction::Transaction(EventTask event) : baseEvent(event)
    {
    }

    EventTask Transaction::setTransactionEvent(EventTask const& event)
    {
        baseEvent += event;
        return baseEvent;
    }

    EventTask Transaction::getTransactionEvent()
    {
        return baseEvent;
    }

    void Transaction::operation(ITask::TaskType operation)
    {
        if(operation == ITask::TASK_DEVICE)
        {
            Manager& manager = Manager::getInstance();

            ITask* baseTask = manager.getITaskIfNotFinished(this->baseEvent.getTaskId());
            if(baseTask != nullptr)
            {
                if(baseTask->getTaskType() == ITask::TASK_DEVICE)
                {
                    /* no blocking is needed */
                    return;
                }
            }
        }
        baseEvent.waitForFinished();
    }

    Queue* Transaction::getComputeDeviceQueue(ITask::TaskType)
    {
        Manager& manager = Manager::getInstance();
        ITask* baseTask = manager.getITaskIfNotFinished(this->baseEvent.getTaskId());

        if(baseTask != nullptr)
        {
            if(baseTask->getTaskType() == ITask::TASK_DEVICE)
            {
                /* DeviceTasks from previous task must be reused to guarantee
                 * that the dependency chain not brake
                 */
                auto* task = static_cast<DeviceTask*>(baseTask);
                return task->getComputeDeviceQueue();
            }
            baseEvent.waitForFinished();
        }
        return Environment<>::get().QueueController().getNextStream();
    }

} // namespace pmacc
