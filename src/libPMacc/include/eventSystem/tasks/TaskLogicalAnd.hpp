/**
 * Copyright 2013-2015 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "eventSystem/tasks/ITask.hpp"
#include "eventSystem/tasks/StreamTask.hpp"
#include "eventSystem/EventSystem.hpp"

namespace PMacc
{

    /**
     * TaskLogicalAnd AND-connects tasks to a new single task
     */
    class TaskLogicalAnd : public StreamTask
    {
    public:

        /**
         * s1 and s1 must be a valid IStreamTask
         * constructor
         */
        TaskLogicalAnd(ITask* s1, ITask* s2) :
        StreamTask(),
        task1(s1->getId()),
        task2(s2->getId())
        {
            combine(s1, s2);
        }

        /*
         * destructor
         */
        virtual ~TaskLogicalAnd()
        {

            notify(this->myId, LOGICALAND, NULL);
        }

        void init()
        {

        }

        bool executeIntern()
        {
            // TaskLogicalAnd is finished if all subtasks are
            // finished (removed) and there is no current work
            // std::cout<<"id1="<<task1<<" id2="<<task2<<std::endl;
            return (task1 == 0) && (task2 == 0);
        }

        void event(id_t eventId, EventType, IEventData*)
        {
            if (task1 == eventId)
            {
                task1 = 0;
                /* \todo: there is a bug in this part of code
                 * ITask* task = Environment<>::get().Manager().getITaskIfNotFinished(task2);
                if (task != NULL)
                {
                    ITask::TaskType type = task->getTaskType();
                    if (type == ITask::TASK_CUDA && this->getTaskType() != ITask::TASK_CUDA)
                    {
                        this->setTaskType(task->getTaskType());
                        this->setCudaEvent(static_cast<StreamTask*> (task)->getCudaEvent());
                    }
                }*/
            } else if (task2 == eventId)
            {
                task2 = 0;
               /* if (task1 != 0)
                {
                    ITask* task = Environment<>::get().Manager().getITaskIfNotFinished(task1);
                    if (task != NULL)
                    {
                        ITask::TaskType type = task->getTaskType();
                        if (type == ITask::TASK_CUDA && this->getTaskType() != ITask::TASK_CUDA)
                        {
                            this->setTaskType(task->getTaskType());
                            this->setCudaEvent(static_cast<StreamTask*> (task)->getCudaEvent());
                        }
                    }
                }*/
            } else
                std::runtime_error("task id not known");

            if(executeIntern())
            {
                delete this;
            }
        }

        std::string toString()
        {
            return "TaskLogicalAnd";
        }

    private:

        inline void combine(ITask* s1, ITask* s2)
        {
            s1->addObserver(this);
            s2->addObserver(this);
            if (s1->getTaskType() == ITask::TASK_CUDA && s2->getTaskType() == ITask::TASK_CUDA)
            {
                this->setTaskType(ITask::TASK_CUDA);
                this->setEventStream(static_cast<StreamTask*> (s2)->getEventStream());
                this->getEventStream()->waitOn(static_cast<StreamTask*> (s1)->getCudaEvent());
                this->activate();
            } else if (s1->getTaskType() == ITask::TASK_MPI && s2->getTaskType() == ITask::TASK_CUDA)
            {
                this->setTaskType(ITask::TASK_MPI);
                this->setEventStream(static_cast<StreamTask*> (s2)->getEventStream());
            } else if (s2->getTaskType() == ITask::TASK_MPI && s1->getTaskType() == ITask::TASK_CUDA)
            {
                this->setTaskType(ITask::TASK_MPI);
                this->setEventStream(static_cast<StreamTask*> (s1)->getEventStream());
            } else if (s1->getTaskType() == ITask::TASK_MPI && s2->getTaskType() == ITask::TASK_MPI)
            {
                this->setTaskType(ITask::TASK_MPI);
            }
        }

        id_t task1;
        id_t task2;
    };

} //namespace PMacc

