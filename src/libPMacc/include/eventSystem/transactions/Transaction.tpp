/**
 * Copyright 2013-2016 Rene Widera, Benjamin Worpitz
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

Transaction::Transaction( EventTask event ) : baseEvent( event )
{

}

inline EventTask Transaction::setTransactionEvent( const EventTask& event )
{
    baseEvent += event;
    return baseEvent;
}

inline EventTask Transaction::getTransactionEvent( )
{
    return baseEvent;
}

void Transaction::operation( ITask::TaskType operation )
{
    if ( operation == ITask::TASK_CUDA )
    {
        Manager &manager = Environment<>::get( ).Manager( );

        ITask* baseTask = manager.getITaskIfNotFinished( this->baseEvent.getTaskId( ) );
        if ( baseTask != NULL )
        {
            if ( baseTask->getTaskType( ) == ITask::TASK_CUDA )
            {
                /* no blocking is needed */
                return;
            }
        }
    }
    baseEvent.waitForFinished( );
}

EventStream* Transaction::getEventStream( ITask::TaskType )
{
    Manager &manager = Environment<>::get( ).Manager( );
    ITask* baseTask = manager.getITaskIfNotFinished( this->baseEvent.getTaskId( ) );

    if ( baseTask != NULL )
    {
        if ( baseTask->getTaskType( ) == ITask::TASK_CUDA )
        {
            /* `StreamTask` from previous task must be reused to guarantee
             * that the dependency chain not brake
             */
            StreamTask* task = static_cast<StreamTask*> ( baseTask );
            return task->getEventStream( );
        }
        baseEvent.waitForFinished( );
    }
    return Environment<>::get( ).StreamController( ).getNextStream( );
}

} //namespace PMacc
