/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz
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
#include "pmacc/eventSystem/tasks/ITask.hpp"
#include "pmacc/eventSystem/tasks/TaskLogicalAnd.hpp"

namespace pmacc
{
    inline EventTask::EventTask(id_t taskId) : taskId(taskId)
    {
    }

    inline EventTask::EventTask() : taskId(0)
    {
    }

    inline std::string EventTask::toString()
    {
        ITask* task = Environment<>::get().Manager().getITaskIfNotFinished(taskId);
        if(task != nullptr)
            return task->toString();

        return std::string();
    }

    inline id_t EventTask::getTaskId() const
    {
        return taskId;
    }

    inline bool EventTask::isFinished()
    {
        return (Environment<>::get().Manager().getITaskIfNotFinished(taskId) == nullptr);
    }

    inline void EventTask::waitForFinished() const
    {
        Environment<>::get().Manager().waitForFinished(taskId);
    }

    inline EventTask EventTask::operator+(const EventTask& other)
    {
        EventTask tmp = *this;
        return tmp += other;
    }

    inline EventTask& EventTask::operator+=(const EventTask& other)
    {
        // If one of the two tasks is already finished, the other task is returned.
        // Otherwise, a TaskLogicalAnd is created and added to the Manager's queue.
        Manager& manager = Environment<>::get().Manager();

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

        TaskLogicalAnd* taskAnd = new TaskLogicalAnd(myTask, otherTask);
        this->taskId = taskAnd->getId();
        manager.addPassiveTask(taskAnd);

        return *this;
    }

    inline EventTask& EventTask::operator=(const EventTask& other)
    {
        // this is faster than a copy constructor
        taskId = other.taskId;
        return *this;
    }

} // namespace pmacc
