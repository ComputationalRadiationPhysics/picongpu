/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Benjamin Worpitz
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
#include "pmacc/eventSystem/transactions/Transaction.hpp"

#include <stack>

namespace pmacc
{
    // forward declaration
    template<unsigned DIM>
    class Environment;

    class EventStream;

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
         * @return an EventStream which can be used for StreamTasks
         */
        void startOperation(ITask::TaskType op);

        /**
         * Adds event to the base event of the top-most transaction.
         *
         * @param event event to add to base event
         * @return new base event
         */
        EventTask setTransactionEvent(const EventTask& event);

        /**
         * Returns the base event of the top-most transaction.
         *
         * @return base event
         */
        EventTask getTransactionEvent();

        EventStream* getEventStream(ITask::TaskType op);

    private:
        friend struct detail::Environment;

        TransactionManager();

        TransactionManager(const TransactionManager& cc);

        static TransactionManager& getInstance();

        std::stack<Transaction> transactions;
    };


} // namespace pmacc
