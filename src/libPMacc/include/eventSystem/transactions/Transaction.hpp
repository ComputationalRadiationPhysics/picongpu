/**
 * Copyright 2013 Felix Schmitt, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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


#ifndef TRANSACTION_HPP
#define	TRANSACTION_HPP

#include "eventSystem/EventSystem.hpp"



namespace PMacc
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
    Transaction(EventTask event, bool isAtomic = false);

    /**
     * Adds event to the base event of this transaction.
     *
     * @param event EventTask to add to base event
     * @return new base event
     */
    EventTask setTransactionEvent(const EventTask& event);

    /**
     * Returns the current base event.
     *
     * @return current base event
     */
    EventTask getTransactionEvent();

    /**
     * Performs an operation on the transaction which leads to synchronization.
     *
     * @param operation type of operation to perform, defines resulting sychronization.
     */
    void operation(ITask::TaskType operation);

    /* Get a EventStream which inlcude all dependencies
     * @param operation type of operation to perform
     * @return EventStream with solved dependencies
     */
    EventStream* getEventStream(ITask::TaskType operation);

private:
    EventTask baseEvent;
    EventStream *eventStream;
    bool isAtomic;

};

}


#endif	/* TRANSACTION_HPP */

