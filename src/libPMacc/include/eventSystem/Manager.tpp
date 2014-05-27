/**
 * Copyright 2013-2014 Felix Schmitt, Rene Widera
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

#include <set>
#include <iostream>

#include "eventSystem/EventSystem.hpp"
#include "eventSystem/streams/StreamController.hpp"
#include "eventSystem/Manager.hpp"
#include <stdlib.h>
#include <stdio.h>
//#define DEBUG_EVENTS

namespace PMacc
{

inline Manager::~Manager( )
{
    CUDA_CHECK( cudaGetLastError( ) );
    waitForAllTasks( );
    CUDA_CHECK( cudaGetLastError( ) );
    delete eventPool;
    CUDA_CHECK( cudaGetLastError( ) );
}

inline bool Manager::execute( id_t taskToWait )
{
#ifdef DEBUG_EVENTS
    static int old_max = 0;
    static int deep = -1;
        static int counter = 0;
    ++counter;

    deep++;
    if ( deep > old_max )
    {
        old_max = deep;
     }
#endif

    static TaskMap::iterator iter = tasks.begin( );

    if ( iter == tasks.end( ) )
        iter = tasks.begin( );

    // this is the slow but very save variant to delete taks in a map
    while ( iter != tasks.end( ) )
    {
        id_t id = iter->first;
        ITask* taskPtr = iter->second;
        assert( taskPtr != NULL );
        ++iter;
#ifdef DEBUG_EVENTS
        if ( counter == 500000 )
            std::cout << taskPtr->toString( ) << " " << passiveTasks.size( ) << std::endl;
#endif
        if ( taskPtr->execute( ) )
        {
            /*test if task is deleted by other stackdeep*/
            if ( getActiveITaskIfNotFinished( id ) == taskPtr )
            {
                tasks.erase( id );
                delete taskPtr;
            }
#ifdef DEBUG_EVENTS
            counter = 0;
#endif

            if ( taskToWait == id )
            {
                iter = tasks.end( );
#ifdef DEBUG_EVENTS
                --deep;
#endif
                return true; //jump out because searched task is finished
            }
        }
    }

#ifdef DEBUG_EVENTS
    --deep;
#endif

    return false;
}

inline void Manager::event( id_t eventId, EventType, IEventData* )
{
    passiveTasks.erase( eventId );
}



inline ITask* Manager::getITaskIfNotFinished( id_t taskId ) const
{
    ITask* passiveTask = getPassiveITaskIfNotFinished( taskId );
    if ( passiveTask != NULL )
        return passiveTask;

    return getActiveITaskIfNotFinished( taskId );
}

inline ITask* Manager::getPassiveITaskIfNotFinished( id_t taskId ) const
{
    TaskMap::const_iterator itPassive = passiveTasks.find( taskId );
    if ( itPassive != passiveTasks.end( ) )
        return itPassive->second;
    return NULL;
}

inline ITask* Manager::getActiveITaskIfNotFinished( id_t taskId ) const
{
    TaskMap::const_iterator it = tasks.find( taskId );
    if ( it != tasks.end( ) )
        return it->second;
    return NULL;
}

inline void Manager::waitForFinished( id_t taskId )
{
    //check if task is passive and wait on it
    ITask* task = getPassiveITaskIfNotFinished( taskId );
    if ( task != NULL )
    {
        do
        {
            this->execute( );
        }
        while ( getPassiveITaskIfNotFinished( taskId ) != NULL );

        return; //we can jump out because task is passive task
    }

    //check if task is  active and wait on it
    task = getActiveITaskIfNotFinished( taskId );
    if ( task != NULL )
    {
        do
        {
            if ( this->execute( taskId ) )
                return; //jump out because task is finished
        }
        while ( getActiveITaskIfNotFinished( taskId ) != NULL );
    }
}

inline void Manager::waitForAllTasks( )
{
    while ( tasks.size( ) != 0 || passiveTasks.size( ) != 0 )
    {
        this->execute( );
    }
    assert( tasks.size( ) == 0 );
}

inline void Manager::addTask( ITask *task )
{
    assert( task != NULL );
    tasks[task->getId( )] = task;
}

inline void Manager::addPassiveTask( ITask *task )
{
    assert( task != NULL );

    task->addObserver( this );
    passiveTasks[task->getId( )] = task;
}

inline Manager::Manager( )
{
    /**
     * The \see Environment ensures that the \see StreamController is
     * already created before calling this
     */
    eventPool = new EventPool( );
    eventPool->addEvents( 300 );
}

inline Manager::Manager( const Manager& )
{
}

inline EventPool& Manager::getEventPool( )
{
    return *eventPool;
}

inline int Manager::getCount( )
{
    for ( TaskMap::iterator iter = tasks.begin( ); iter != tasks.end( ); ++iter )
    {
        if ( iter->second != NULL )
        {
            std::cout << iter->first << " = " << iter->second->toString( ) << std::endl;
        }
    }
    return tasks.size( );
}




}

