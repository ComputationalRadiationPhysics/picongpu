/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "memory/buffers/Exchange.hpp"

#include "eventSystem/tasks/Factory.hpp"
#include "eventSystem/tasks/ITask.hpp"

namespace PMacc
{

    /**
     * Singleton Factory-pattern class for creation of several types of EventTasks.
     * Tasks are not actually 'returned' but immediately initialised and
     * added to the Manager's queue. An exception is TaskKernel.
     */
    class FieldFactory
    {
    public:

        /**
         * Creates a TaskReceive.
         * @param ex Exchange to create new TaskReceive with
         * @param task_out returns the newly created task
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template<class Field>
        EventTask createTaskFieldReceiveAndInsert(Field &buffer,
        ITask *registeringTask = NULL);

        template<class Field>
        EventTask createTaskFieldReceiveAndInsertExchange(Field &buffer, uint32_t exchange,
        ITask *registeringTask = NULL);

        /**
         * Creates a TaskSend.
         * @param ex Exchange to create new TaskSend with
         * @param task_in TaskReceive to register at new TaskSend
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an observer
         */
        template<class Field>
        EventTask createTaskFieldSend(Field &buffer,
        ITask *registeringTask = NULL);

        template<class Field>
        EventTask createTaskFieldSendExchange(Field &buffer, uint32_t exchange,
        ITask *registeringTask = NULL);

        /**
         * returns the instance of this factory
         * @return the instance
         */
        static FieldFactory& getInstance()
        {
            static FieldFactory instance;
            return instance;
        }

    private:

        FieldFactory() { };

        FieldFactory(const FieldFactory&) { };

    };

} //namespace PMacc

#include "fields/tasks/FieldFactory.tpp"

