/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/events/EventTask.hpp"
#include "pmacc/eventSystem/tasks/ITask.hpp"

namespace pmacc
{
    class Queue;

    /**
     * Represents a single transaction in the task/event synchronization system.
     */
    class Transaction
    {
    public:
        /**
         * Constructor.
         *
         * @param event initial EventTask for base event
         */
        Transaction(EventTask event);

        /**
         * Adds event to the base event of this transaction.
         *
         * @param event EventTask to add to base event
         * @return new base event
         */
        EventTask setTransactionEvent(EventTask const& event);

        /**
         * Returns the current base event.
         *
         * @return current base event
         */
        EventTask getTransactionEvent();

        /**
         * Performs an operation on the transaction which leads to synchronization.
         *
         * @param operation type of operation to perform, defines resulting synchronization.
         */
        void operation(ITask::TaskType operation);

        /* Get a Queue which include all dependencies
         * @param operation type of operation to perform
         * @return Queue with solved dependencies
         */
        Queue* getComputeDeviceQueue(ITask::TaskType operation);

    private:
        EventTask baseEvent;
    };

} // namespace pmacc
