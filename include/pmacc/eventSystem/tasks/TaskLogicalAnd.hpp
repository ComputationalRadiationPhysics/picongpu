/* Copyright 2013-2023 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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


#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/tasks/DeviceTask.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"

namespace pmacc
{
    /**
     * TaskLogicalAnd AND-connects tasks to a new single task
     */
    class TaskLogicalAnd : public DeviceTask
    {
    public:
        /**
         * s1 and s1 must be a valid DeviceTasks
         * constructor
         */
        TaskLogicalAnd(ITask* s1, ITask* s2) : DeviceTask(), task1(s1->getId()), task2(s2->getId())
        {
            combine(s1, s2);
        }

        /*
         * destructor
         */
        ~TaskLogicalAnd() override
        {
            notify(this->myId, LOGICALAND, nullptr);
        }

        void init() override
        {
        }

        bool executeIntern() override
        {
            /*  TaskLogicalAnd is finished if all subtasks are
             *  finished (removed) and there is no current work
             */
            return (task1 == 0) && (task2 == 0);
        }

        void event(id_t eventId, EventType, IEventData*) override
        {
            if(task1 == eventId)
            {
                task1 = 0;

                ITask* task = Manager::getInstance().getITaskIfNotFinished(task2);
                if(task != nullptr)
                {
                    ITask::TaskType type = task->getTaskType();
                    if(type == ITask::TASK_DEVICE)
                    {
                        this->stream = static_cast<DeviceTask*>(task)->getComputeDeviceQueue();
                        this->setTaskType(ITask::TASK_DEVICE);
                        this->m_alpakaEvent = static_cast<DeviceTask*>(task)->getComputeEventHandle();
                        this->hasComputeEventHandle = true;
                    }
                }
            }
            else if(task2 == eventId)
            {
                task2 = 0;

                ITask* task = Manager::getInstance().getITaskIfNotFinished(task1);
                if(task != nullptr)
                {
                    ITask::TaskType type = task->getTaskType();
                    if(type == ITask::TASK_DEVICE)
                    {
                        this->stream = static_cast<DeviceTask*>(task)->getComputeDeviceQueue();
                        this->setTaskType(ITask::TASK_DEVICE);
                        this->m_alpakaEvent = static_cast<DeviceTask*>(task)->getComputeEventHandle();
                        this->hasComputeEventHandle = true;
                    }
                }
            }
            else
                std::runtime_error("task id not known");

            if(executeIntern())
            {
                delete this;
            }
        }

        std::string toString() override
        {
            return std::string("TaskLogicalAnd (") + EventTask(task1).toString() + std::string(" - ")
                + EventTask(task2).toString() + std::string(" )");
        }

    private:
        inline void combine(ITask* s1, ITask* s2)
        {
            s1->addObserver(this);
            s2->addObserver(this);
            if(s1->getTaskType() == ITask::TASK_DEVICE && s2->getTaskType() == ITask::TASK_DEVICE)
            {
                this->setTaskType(ITask::TASK_DEVICE);
                this->setQueue(static_cast<DeviceTask*>(s2)->getComputeDeviceQueue());
                if(static_cast<DeviceTask*>(s1)->getComputeDeviceQueue()
                   != static_cast<DeviceTask*>(s2)->getComputeDeviceQueue())
                    this->getComputeDeviceQueue()->waitOn(static_cast<DeviceTask*>(s1)->getComputeEventHandle());
                this->activate();
            }
            else if(s1->getTaskType() == ITask::TASK_MPI && s2->getTaskType() == ITask::TASK_DEVICE)
            {
                this->setTaskType(ITask::TASK_MPI);
            }
            else if(s2->getTaskType() == ITask::TASK_MPI && s1->getTaskType() == ITask::TASK_DEVICE)
            {
                this->setTaskType(ITask::TASK_MPI);
            }
            else if(s1->getTaskType() == ITask::TASK_MPI && s2->getTaskType() == ITask::TASK_MPI)
            {
                this->setTaskType(ITask::TASK_MPI);
            }
        }

        id_t task1;
        id_t task2;
    };

} // namespace pmacc
