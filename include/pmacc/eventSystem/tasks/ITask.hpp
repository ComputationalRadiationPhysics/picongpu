/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/assert.hpp"
#include "pmacc/eventSystem/events/EventNotify.hpp"
#include "pmacc/eventSystem/events/IEvent.hpp"
#include "pmacc/types.hpp"

#include <set>
#include <string>

namespace pmacc
{
    /**
     * Abstract base class for all tasks.
     */
    class ITask
        : public EventNotify
        , public IEvent
    {
    public:
        enum TaskType
        {
            TASK_UNKNOWN,
            TASK_DEVICE,
            TASK_MPI,
            TASK_HOST
        };

        /**
         * constructor
         */
        ITask()
        {
            // task id 0 is reserved for invalid
            static id_t globalId = 1;

            myId = globalId++;
            PMACC_ASSERT(myId > 0);
        }

        ~ITask() override = default;

        /**
         * Executes this task.
         *
         * @return true if the task is finished, false otherwise.
         */
        bool execute()
        {
            // std::cout << "execute: " << toString() << std::endl;
            return executeIntern();
        }

        /**
         * Initializes the task.
         * Must be called before adding the task to the Manager's queue.
         */
        virtual void init() = 0;

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
        TaskType myType{ITask::TASK_UNKNOWN};
    };

} // namespace pmacc
