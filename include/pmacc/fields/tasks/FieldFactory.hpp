/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"
#include "pmacc/memory/buffers/Exchange.hpp"

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
        FieldFactory() = default;

        FieldFactory(FieldFactory const&) = default;
    };

} // namespace pmacc
