/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/eventSystem/events/EventTask.hpp"

namespace pmacc
{
    class EventStream;

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
        HINLINE Transaction(EventTask event);

        /**
         * Adds event to the base event of this transaction.
         *
         * @param event EventTask to add to base event
         * @return new base event
         */
        HINLINE EventTask setTransactionEvent(const EventTask& event);

        /**
         * Returns the current base event.
         *
         * @return current base event
         */
        HINLINE EventTask getTransactionEvent();

        /**
         * Performs an operation on the transaction which leads to synchronization.
         *
         * @param operation type of operation to perform, defines resulting synchronization.
         */
        HINLINE void operation(ITask::TaskType operation);

        /* Get a EventStream which include all dependencies
         * @param operation type of operation to perform
         * @return EventStream with solved dependencies
         */
        HINLINE EventStream* getEventStream(ITask::TaskType operation);

    private:
        EventTask baseEvent;
    };

} // namespace pmacc
