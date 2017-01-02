/**
 * Copyright 2016 Rene Widera
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

#include "pmacc_types.hpp"
#include "eventSystem/events/CudaEventHandle.hpp"
#include "eventSystem/events/CudaEvent.hpp"
#include <cuda_runtime.h>

namespace PMacc
{
    CudaEvent::CudaEvent( ) : isRecorded( false ), finished( true ), refCounter( 0u )
    {
        log( ggLog::CUDA_RT()+ggLog::MEMORY(), "create event" );
        CUDA_CHECK( cudaEventCreateWithFlags( &event, cudaEventDisableTiming ) );
    }


    CudaEvent::~CudaEvent( )
    {
        PMACC_ASSERT( refCounter == 0u );
        log( ggLog::CUDA_RT()+ggLog::MEMORY(), "sync and delete event" );
        // free cuda event
        CUDA_CHECK( cudaEventSynchronize( event ) );
        CUDA_CHECK( cudaEventDestroy( event ) );

    }

    void CudaEvent::registerHandle()
    {
        ++refCounter;
    }

    void CudaEvent::releaseHandle()
    {
        assert( refCounter != 0u );
        // get old value and decrement
        uint32_t oldCounter = refCounter--;
        if( oldCounter == 1u )
        {
            // reset event meta data
            isRecorded = false;
            finished = true;

            Environment<>::get().EventPool( ).push( this );
        }
    }


    bool CudaEvent::isFinished()
    {
        // avoid cuda driver calls if event is already finished
        if( finished )
            return true;
        assert( isRecorded );

        cudaError_t rc = cudaEventQuery(event);

        if(rc == cudaSuccess)
        {
            finished = true;
            return true;
        }
        else if(rc == cudaErrorNotReady)
            return false;
        else
            PMACC_PRINT_CUDA_ERROR_AND_THROW(rc, "Event query failed");
    }


    void CudaEvent::recordEvent(cudaStream_t stream)
    {
        /* disallow double recording */
        assert(isRecorded == false);
        isRecorded = true;
        finished = false;
        this->stream = stream;
        CUDA_CHECK(cudaEventRecord(event, stream));
    }

} // namepsace PMacc
