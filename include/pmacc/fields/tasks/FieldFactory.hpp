/* Copyright 2013-2021 Rene Widera
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

#include "pmacc/memory/buffers/Exchange.hpp"

#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"

namespace pmacc
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
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an
         * observer
         */
        template<class Field>
        EventTask createTaskFieldReceiveAndInsert(Field& buffer, ITask* registeringTask = nullptr);

        template<class Field>
        EventTask createTaskFieldReceiveAndInsertExchange(
            Field& buffer,
            uint32_t exchange,
            ITask* registeringTask = nullptr);

        /**
         * Creates a TaskSend.
         * @param ex Exchange to create new TaskSend with
         * @param task_in TaskReceive to register at new TaskSend
         * @param registeringTask optional pointer to an ITask which should be registered at the new task as an
         * observer
         */
        template<class Field>
        EventTask createTaskFieldSend(Field& buffer, ITask* registeringTask = nullptr);

        template<class Field>
        EventTask createTaskFieldSendExchange(Field& buffer, uint32_t exchange, ITask* registeringTask = nullptr);

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
        FieldFactory(){};

        FieldFactory(const FieldFactory&){};
    };

} // namespace pmacc

#include "pmacc/fields/tasks/FieldFactory.tpp"
