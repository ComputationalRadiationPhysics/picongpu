/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pmacc/eventSystem/events/EventTask.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"
#include "pmacc/eventSystem/tasks/TaskLogicalAnd.hpp"

namespace pmacc
{
    EventTask::EventTask(id_t taskId) : taskId(taskId)
    {
    }

    std::string EventTask::toString()
    {
        ITask* task = Manager::getInstance().getITaskIfNotFinished(taskId);
        if(task != nullptr)
            return task->toString();

        return std::string();
    }

    bool EventTask::isFinished()
    {
        return (Manager::getInstance().getITaskIfNotFinished(taskId) == nullptr);
    }

    id_t EventTask::getTaskId() const
    {
        return taskId;
    }

    void EventTask::waitForFinished() const
    {
        Manager::getInstance().waitForFinished(taskId);
    }

    EventTask EventTask::operator+(EventTask const& other)
    {
        EventTask tmp = *this;
        return tmp += other;
    }

    EventTask& EventTask::operator+=(EventTask const& other)
    {
        // If one of the two tasks is already finished, the other task is returned.
        // Otherwise, a TaskLogicalAnd is created and added to the Manager's queue.
        Manager& manager = Manager::getInstance();

        if(this->taskId == other.taskId)
            return *this;

        ITask* myTask = manager.getITaskIfNotFinished(this->taskId);
        if(myTask == nullptr)
        {
            this->taskId = other.taskId;
            return *this;
        }

        ITask* otherTask = manager.getITaskIfNotFinished(other.taskId);
        if(otherTask == nullptr)
        {
            return *this;
        }

        auto* taskAnd = new TaskLogicalAnd(myTask, otherTask);
        this->taskId = taskAnd->getId();
        manager.addPassiveTask(taskAnd);

        return *this;
    }

} // namespace pmacc
