/**
 * Copyright 2013-2015 Rene Widera, Benjamin Worpitz
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

#include "eventSystem/transactions/Transaction.hpp"

#include "eventSystem/streams/StreamController.hpp"
#include "eventSystem/events/EventTask.hpp"
#include "eventSystem/tasks/StreamTask.hpp"

namespace PMacc
{

Transaction::Transaction( EventTask event, bool isAtomic ) : baseEvent( event ), eventStream( NULL ), isAtomic( isAtomic )
{
    eventStream = Environment<>::get().StreamController().getNextStream( );
    event.waitForFinished( );
}

inline EventTask Transaction::setTransactionEvent( const EventTask& event )
{
    Manager &manager = Environment<>::get().Manager();
    ITask* baseTask = manager.getITaskIfNotFinished( event.getTaskId( ) );

    if ( baseTask != NULL )
    {
        if ( baseTask->getTaskType( ) == ITask::TASK_CUDA )
        {
            StreamTask* task = static_cast<StreamTask*> ( baseTask );
            this->eventStream->waitOn(task->getCudaEvent( ));
        }
    }

    baseEvent += event;
    return baseEvent;
}

inline EventTask Transaction::getTransactionEvent( )
{
    return baseEvent;
}

void Transaction::operation( ITask::TaskType )
{
    if ( isAtomic == false )
        baseEvent.waitForFinished( );
}

EventStream* Transaction::getEventStream( ITask::TaskType )
{
    return this->eventStream;
}

} //namespace PMacc
