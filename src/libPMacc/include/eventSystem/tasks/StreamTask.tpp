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

#include "Environment.hpp"
//#include "eventSystem/EventSystem.hpp"
#include "eventSystem/tasks/StreamTask.hpp"
#include "eventSystem/streams/EventStream.hpp"
#include "assert.hpp"

namespace PMacc
{

inline StreamTask::StreamTask( ) :
ITask( ),
stream( NULL ),
hasCudaEventHandle( false ),
alwaysFinished( false )
{
    this->setTaskType( ITask::TASK_CUDA );
}

inline CudaEventHandle StreamTask::getCudaEventHandle( ) const
{
    PMACC_ASSERT( hasCudaEventHandle );
    return cudaEvent;
}

inline void StreamTask::setCudaEventHandle(const CudaEventHandle& cudaEvent )
{
    this->hasCudaEventHandle = true;
    this->cudaEvent = cudaEvent;
}

inline bool StreamTask::isFinished( )
{
    if ( alwaysFinished )
        return true;
    if ( hasCudaEventHandle )
    {
        if ( cudaEvent.isFinished( ) )
        {
            alwaysFinished = true;
            return true;
        }
    }
    return false;
}

inline EventStream* StreamTask::getEventStream( )
{
    if ( stream == NULL )
        stream = __getEventStream( TASK_CUDA );
    return stream;
}

inline void StreamTask::setEventStream( EventStream* newStream )
{
    PMACC_ASSERT( newStream != NULL );
    PMACC_ASSERT( stream == NULL ); //it is only allowed to set a stream if no stream is set before
    this->stream = newStream;
}

inline cudaStream_t StreamTask::getCudaStream( )
{
    if ( stream == NULL )
        stream = Environment<>::get( ).TransactionManager( ).getEventStream( TASK_CUDA );
    return stream->getCudaStream( );
}

inline void StreamTask::activate( )
{
    cudaEvent = Environment<>::get().EventPool( ).pop( );
    cudaEvent.recordEvent(this->stream->getCudaStream());
    hasCudaEventHandle = true;
}

} //namespace PMacc
