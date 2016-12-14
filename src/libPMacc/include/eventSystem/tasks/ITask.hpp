/**
 * Copyright 2013-2016 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
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

#include "eventSystem/events/EventNotify.hpp"
#include "eventSystem/events/IEvent.hpp"
#include "pmacc_types.hpp"
#include "assert.hpp"

#include <string>
#include <set>


namespace PMacc
{
    /**
     * Abstract base class for all tasks.
     */
    class ITask : public EventNotify, public IEvent
    {
    public:

        enum TaskType
        {
            TASK_UNKNOWN, TASK_CUDA, TASK_MPI, TASK_HOST
        };

        /**
         * constructor
         */
        ITask(): myType(ITask::TASK_UNKNOWN)
        {
            // task id 0 is reserved for invalid
            static id_t globalId = 1;

            myId = globalId++;
            PMACC_ASSERT(myId > 0);
        }


        virtual ~ITask()
        {
        }

        /**
         * Executes this task.
         *
         * @return true if the task is finished, false otherwise.
         */
        bool execute()
        {
            //std::cout << "execute: " << toString() << std::endl;
            return executeIntern();
        }

        /**
         * Initializes the task.
         * Must be called before adding the task to the Manager's queue.
         */
        virtual void init()=0;

        /**
         * Returns the unique id of this task.
         * If two tasks have the same id, they are the same task for the manager.
         *
         * @return the task id
         */
        id_t getId() const
        {
            return myId;
        }

        /**
         * Returns the type of the task.
         *
         * @return the task type
         */
        virtual ITask::TaskType getTaskType()
        {
            return myType;
        }

        /**
         * Sets the type of the task.
         *
         * @param newType new task type
         */
        void setTaskType(ITask::TaskType newType)
        {
            myType = newType;
        }

        /**
         * Returns a string representation of the task.
         *
         * @return a string naming this task
         */
        virtual std::string toString() = 0;
    protected:
        virtual bool executeIntern() = 0;

        id_t myId;
        TaskType myType;
    };

} //namespace PMacc
