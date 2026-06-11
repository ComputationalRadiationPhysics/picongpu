/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/events/EventTask.hpp"
#include "pmacc/eventSystem/transactions/Transaction.hpp"

#include <stack>

namespace pmacc
{
    /**
     * Manages the task/event synchronization system using task 'transactions'.
     * Transactions are grouped on a stack.
     */
    class TransactionManager
    {
    public:
        /**
         * Destructor.
         */
        virtual ~TransactionManager() /*noexcept(false)*/;

        /**
         * Adds a new transaction to the stack.
         *
         * @param serialEvent initial base event for new transaction
         */
        void startTransaction(EventTask serialEvent = EventTask());

        /**
         * Removes the top-most transaction from the stack.
         *
         * @return the base event of the removed transaction
         */
        EventTask endTransaction();

        /**
         * Synchronizes a blocking operation with events on the top-most transaction.
         *
         * @param op operation type for synchronization
         */
        void startOperation(ITask::TaskType op);

        /**
         * Adds event to the base event of the top-most transaction.
         *
         * @param event event to add to base event
         * @return new base event
         */
        EventTask setTransactionEvent(EventTask const& event);

        /**
         * Returns the base event of the top-most transaction.
         *
         * @return base event
         */
        EventTask getTransactionEvent();

        Queue* getComputeDeviceQueue(ITask::TaskType op);

        static TransactionManager& getInstance()
        {
            static TransactionManager instance;
            return instance;
        }

        TransactionManager(TransactionManager const& cc) = delete;

    private:
        TransactionManager();

        std::stack<Transaction> transactions;
    };


} // namespace pmacc
